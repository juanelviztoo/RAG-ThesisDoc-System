"""
src/rag/self_query.py
=====================
V3.0b — Metadata Self-Querying Filtering

Tanggung jawab:
    - Menerima user query (bahasa alami) + cfg
    - LLM mengekstrak filter metadata terstruktur via Few-Shot Prompting
    - Mengembalikan SelfQueryResult: ChromaDB where clause + semantic_query
    - Graceful fallback: jika LLM gagal / filter tidak valid → filters=None
        (caller di app_streamlit.py wajib fallback ke retrieval penuh)

Public API:
    build_self_query(cfg, query) → SelfQueryResult
    check_filter_has_results(where_filter, cfg, collection) → bool

Prinsip desain:
    - Verbose & loggable : SelfQueryResult membawa semua data untuk BAB IV
    - Conservative       : hanya filter field yang high-confidence
    - Graceful           : TIDAK PERNAH melempar exception ke caller
    - Audit-ready        : llm_raw_output + extracted_metadata tersedia untuk debug

Catatan implementasi (ChromaDB where clause):
    - tahun  : operator "$eq"       — integer exact match (field terpercaya)
    - prodi  : operator "$contains" — substring match (toleran variasi format cover)
    - penulis: operator "$contains" — substring match (toleran variasi penulisan nama)

    Format 1 kondisi : {"field": {"$operator": value}}
    Format 2+ kondisi: {"$and": [{"f1": {..}}, {"f2": {..}}]}
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


# ═════════════════════════════════════════════════════════════════════════════
# Konstanta
# ═════════════════════════════════════════════════════════════════════════════

_VALID_YEAR_MIN = 2000
_VALID_YEAR_MAX = 2030

# Field metadata yang bisa difilter (urutan = prioritas aplikasi filter)
# Prioritas final V3.0b: tahun > prodi > penulis
# Dengan max_filter_fields=2, kombinasi maksimal yang dipakai adalah:
#   - tahun + prodi
#   - tahun + penulis
#   - prodi + penulis
# sesuai urutan kemunculan field valid setelah proses validasi.
_FILTER_FIELDS = ("tahun", "prodi", "penulis")

# Nilai placeholder dari pdf_parser.py yang menandakan metadata tidak terdeteksi
# → field dengan nilai ini TIDAK BOLEH masuk where clause (akan hasilkan 0 hasil)
_UNDETECTED_VALUES = frozenset({
    "",
    "penulis tidak terdeteksi",
    "judul tidak terdeteksi",
    "tidak terdeteksi",
    "unknown",
    "none",
    "n/a",
    "-",
})


# ═════════════════════════════════════════════════════════════════════════════
# Data class — SelfQueryResult
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SelfQueryResult:
    """
    Hasil self-querying: filter metadata + query semantik untuk retrieval.

    Attributes:
        semantic_query     : query untuk embedding retrieval (bagian topik/konten,
                            tanpa informasi metadata seperti tahun/prodi/penulis)
        filters            : ChromaDB where clause dict, atau None.
                            None = tidak ada filter valid → caller wajib fallback
                            ke retrieval penuh tanpa filter.
        extracted_metadata : dict field mentah hasil ekstraksi LLM (untuk logging
                            & analisis di BAB IV skripsi)
        filter_applied     : True jika ada ≥1 filter valid yang akan dipakai
        fallback_reason    : string penjelasan kenapa filters=None (None jika
                            filter_applied=True)
        llm_raw_output     : raw string output LLM sebelum di-parse (untuk debug;
                            tidak di-log default karena bisa panjang)
        filter_fields_used : list nama field yang masuk ke where clause
    """
    semantic_query: str
    filters: Optional[Dict[str, Any]]
    extracted_metadata: Dict[str, Any]
    filter_applied: bool
    fallback_reason: Optional[str]
    llm_raw_output: Optional[str]
    filter_fields_used: List[str]

    def to_log_dict(self) -> Dict[str, Any]:
        """
        Serialize ke dict untuk JSONL logging di app_streamlit.py.
        llm_raw_output tidak disertakan default (bisa panjang).
        """
        return {
            "semantic_query": self.semantic_query,
            "filter_applied": self.filter_applied,
            "filter_fields_used": self.filter_fields_used,
            "filters": self.filters,
            "extracted_metadata": self.extracted_metadata,
            "fallback_reason": self.fallback_reason,
        }


# ═════════════════════════════════════════════════════════════════════════════
# Few-shot examples — domain skripsi akademik Indonesia
# ═════════════════════════════════════════════════════════════════════════════
#
# Format output JSON yang diharapkan dari LLM:
# {
#   "semantic_query": "<topik/konten query, tanpa info metadata>",
#   "tahun"         : <int|null>,
#   "prodi"         : "<string|null>",
#   "penulis"       : "<string|null>"
# }
#
# Aturan few-shot:
# - Jika metadata tidak disebut → null (bukan string "null")
# - semantic_query = bagian TOPIK dari query (dipakai untuk embedding search)
# - tahun  : integer, bukan string; null jika tidak disebut
# - prodi  : nama program studi/jurusan sebagaimana disebut user
# - penulis: nama orang saja (bukan NIM/nomor)

_FEW_SHOT_EXAMPLES = [
    # ── Kasus 1: filter tahun saja ─────────────────────────────────────────
    {
        "query": "Cari skripsi tahun 2023 tentang CNN image classification",
        "output": json.dumps({
            "semantic_query": "CNN image classification",
            "tahun": 2023,
            "prodi": None,
            "penulis": None,
        }, ensure_ascii=False),
    },
    # ── Kasus 2: filter prodi saja ─────────────────────────────────────────
    {
        "query": "Skripsi prodi Informatika yang membahas sistem informasi manajemen",
        "output": json.dumps({
            "semantic_query": "sistem informasi manajemen",
            "tahun": None,
            "prodi": "Informatika",
            "penulis": None,
        }, ensure_ascii=False),
    },
    # ── Kasus 3: filter penulis saja ──────────────────────────────────────
    {
        "query": "Tugas akhir karya Budi Santoso tentang pengembangan aplikasi mobile",
        "output": json.dumps({
            "semantic_query": "pengembangan aplikasi mobile",
            "tahun": None,
            "prodi": None,
            "penulis": "Budi Santoso",
        }, ensure_ascii=False),
    },
    # ── Kasus 4: filter kombinasi tahun + prodi ────────────────────────────
    {
        "query": "Skripsi 2022 jurusan Informatika tentang machine learning klasifikasi teks",
        "output": json.dumps({
            "semantic_query": "machine learning klasifikasi teks",
            "tahun": 2022,
            "prodi": "Informatika",
            "penulis": None,
        }, ensure_ascii=False),
    },
    # ── Kasus 5: filter kombinasi tahun + penulis ──────────────────────────
    {
        "query": "Tugas akhir 2021 oleh Dewi Rahayu tentang IoT sensor suhu",
        "output": json.dumps({
            "semantic_query": "IoT sensor suhu",
            "tahun": 2021,
            "prodi": None,
            "penulis": "Dewi Rahayu",
        }, ensure_ascii=False),
    },
    # ── Kasus 6: filter kombinasi lengkap ─────────────────────────────────
    {
        "query": "Skripsi 2023 Sains Data karya Ahmad Fauzi tentang deteksi anomali",
        "output": json.dumps({
            "semantic_query": "deteksi anomali",
            "tahun": 2023,
            "prodi": "Sains Data",
            "penulis": "Ahmad Fauzi",
        }, ensure_ascii=False),
    },
    # ── Kasus 7: tidak ada metadata → semua null ───────────────────────────
    {
        "query": "Apa metode pengembangan sistem yang digunakan?",
        "output": json.dumps({
            "semantic_query": "metode pengembangan sistem yang digunakan",
            "tahun": None,
            "prodi": None,
            "penulis": None,
        }, ensure_ascii=False),
    },
    # ── Kasus 8: query tahapan metodologi → tidak ada metadata ────────────
    {
        "query": "Jelaskan tahapan RAD pada sistem monitoring",
        "output": json.dumps({
            "semantic_query": "tahapan RAD sistem monitoring",
            "tahun": None,
            "prodi": None,
            "penulis": None,
        }, ensure_ascii=False),
    },
    # ── Kasus 9: query hasil pengujian + filter tahun + prodi ─────────────
    {
        "query": "Hasil pengujian sistem e-ticketing 2023 prodi Sains Data",
        "output": json.dumps({
            "semantic_query": "hasil pengujian sistem e-ticketing",
            "tahun": 2023,
            "prodi": "Sains Data",
            "penulis": None,
        }, ensure_ascii=False),
    },
    # ── Kasus 10: query umum tanpa metadata ───────────────────────────────
    {
        "query": "Skripsi tentang deteksi wajah menggunakan deep learning",
        "output": json.dumps({
            "semantic_query": "deteksi wajah deep learning",
            "tahun": None,
            "prodi": None,
            "penulis": None,
        }, ensure_ascii=False),
    },
]


def _build_few_shot_text() -> str:
    """
    Bangun blok teks few-shot examples untuk system prompt.

    CATATAN PENTING: LangChain ChatPromptTemplate menginterpretasikan {text} sebagai
    template variable. Karena few-shot examples mengandung JSON dengan kurung
    kurawal (mis. {"semantic_query": ...}), semua kurung kurawal di-escape
    dengan cara digandakan {{ dan }} agar LangChain memperlakukannya sebagai
    teks literal, bukan template variable.
    """
    lines: List[str] = []
    for ex in _FEW_SHOT_EXAMPLES:
        lines.append(f'Query: "{ex["query"]}"')
        # Escape { → {{ dan } → }} agar tidak dikira template variable oleh LangChain
        escaped_output = ex["output"].replace("{", "{{").replace("}", "}}")
        lines.append(f"Output: {escaped_output}")
        lines.append("")
    return "\n".join(lines).strip()


# ═════════════════════════════════════════════════════════════════════════════
# LLM builder
# ═════════════════════════════════════════════════════════════════════════════

def _build_llm_for_self_query(cfg: Dict[str, Any]) -> Any:
    """
    Build LLM khusus untuk self-querying.

    Override dari konfigurasi umum:
        - temperature = 0.0  (deterministik — output JSON harus konsisten)
        - max_tokens  = 256  (output JSON pendek, tidak perlu besar)

    Reuse pola _build_llm() dari generate_utils.py untuk konsistensi provider.
    """
    load_dotenv()

    provider_raw = (
        os.getenv("LLM_PROVIDER")
        or (cfg.get("llm") or {}).get("provider", "groq")
        or "groq"
    )
    provider = str(provider_raw).strip().lower()

    # Ambil override dari config self_query, fallback ke nilai default ketat
    sq_cfg = cfg.get("self_query") or {}
    temperature  = float(sq_cfg.get("llm_temperature", 0.0))
    max_tokens   = int(sq_cfg.get("llm_max_tokens", 256))
    timeout      = int((cfg.get("llm") or {}).get("timeout", 60))
    max_retries  = int((cfg.get("llm") or {}).get("max_retries", 2))

    if provider == "groq":
        from langchain_groq import ChatGroq  # type: ignore
        api_key = os.getenv("GROQ_API_KEY")
        model = (
            os.getenv("GROQ_MODEL")
            or (cfg.get("llm") or {}).get("model")
            or "llama-3.1-8b-instant"
        )
        if not api_key:
            raise ValueError("GROQ_API_KEY belum di-set.")
        return ChatGroq(
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI  # type: ignore
        api_key = os.getenv("OPENAI_API_KEY")
        model = (
            os.getenv("OPENAI_MODEL")
            or (cfg.get("llm") or {}).get("model")
            or "gpt-4o-mini"
        )
        if not api_key:
            raise ValueError("OPENAI_API_KEY belum di-set.")
        try:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
                model_kwargs={"max_tokens": max_tokens},
            )
        except TypeError:
            return ChatOpenAI()

    if provider == "ollama":
        from langchain_ollama import ChatOllama  # type: ignore
        base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        model = (
            os.getenv("OLLAMA_MODEL")
            or (cfg.get("llm") or {}).get("model")
            or "llama3"
        )
        return ChatOllama(
            base_url=base_url,
            model=model,
            temperature=temperature,
        )

    raise ValueError(
        f"Provider tidak dikenal: {provider}. Gunakan: groq | openai | ollama"
    )


# ═════════════════════════════════════════════════════════════════════════════
# JSON parsing & validasi
# ═════════════════════════════════════════════════════════════════════════════

def _extract_json_from_llm_output(raw: Optional[str]) -> Optional[str]:
    """
    Ekstrak string JSON dari output LLM yang mungkin mengandung teks tambahan.
    Menerima Optional[str] agar caller tidak perlu type narrowing eksplisit.
    Menangani tiga format umum:
        1. Markdown code fence: ```json { ... } ```
        2. JSON murni langsung: { ... }
        3. JSON di tengah teks: "...teks... { ... } ...teks..."
    """
    if not raw:
        return None

    stripped = raw.strip()

    # Prioritas 1: markdown code fence ```json ... ``` atau ``` ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", stripped)
    if m:
        return m.group(1).strip()

    # Prioritas 2: cari blok { ... } pertama
    # Gunakan greedy dari { pertama sampai } terakhir (untuk nested JSON)
    start = stripped.find("{")
    end   = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1].strip()

    return None


def _validate_and_clean_extracted(raw_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validasi dan bersihkan dict hasil parsing JSON.
    Return dict yang hanya berisi field yang valid dan bernilai bermakna.

    Aturan validasi per field:
        semantic_query : string; jika kosong akan di-fallback ke query asli di caller
        tahun          : integer dalam range [_VALID_YEAR_MIN, _VALID_YEAR_MAX];
                        null jika tidak valid atau tidak ada
        prodi          : string panjang > 2 dan bukan placeholder;
                        null jika tidak valid
        penulis        : string panjang > 2 dan bukan placeholder;
                        null jika tidak valid
    """
    cleaned: Dict[str, Any] = {}

    # ── semantic_query ─────────────────────────────────────────────────────────
    # LLM kadang menggunakan key "query" bukan "semantic_query"
    sq_raw = raw_dict.get("semantic_query") or raw_dict.get("query") or ""
    cleaned["semantic_query"] = str(sq_raw).strip()

    # ── tahun ──────────────────────────────────────────────────────────────────
    tahun_raw = raw_dict.get("tahun")
    if tahun_raw is not None and str(tahun_raw).strip().lower() not in ("null", "none", ""):
        try:
            tahun_int = int(float(str(tahun_raw)))
            if _VALID_YEAR_MIN <= tahun_int <= _VALID_YEAR_MAX:
                cleaned["tahun"] = tahun_int
            else:
                # Tahun di luar range → abaikan, catat sebagai invalid
                cleaned["tahun"] = None
                cleaned["_tahun_invalid"] = tahun_raw  # untuk logging/debug
        except (TypeError, ValueError):
            cleaned["tahun"] = None
    else:
        cleaned["tahun"] = None

    # ── prodi ──────────────────────────────────────────────────────────────────
    prodi_raw = raw_dict.get("prodi")
    if (
        prodi_raw
        and isinstance(prodi_raw, str)
        and str(prodi_raw).strip().lower() not in ("null", "none")
    ):
        p = prodi_raw.strip()
        if p.lower() not in _UNDETECTED_VALUES and len(p) > 2:
            cleaned["prodi"] = p
        else:
            cleaned["prodi"] = None
    else:
        cleaned["prodi"] = None

    # ── penulis ────────────────────────────────────────────────────────────────
    penulis_raw = raw_dict.get("penulis")
    if (
        penulis_raw
        and isinstance(penulis_raw, str)
        and str(penulis_raw).strip().lower() not in ("null", "none")
    ):
        pn = penulis_raw.strip()
        if pn.lower() not in _UNDETECTED_VALUES and len(pn) > 2:
            cleaned["penulis"] = pn
        else:
            cleaned["penulis"] = None
    else:
        cleaned["penulis"] = None

    return cleaned


def _build_chromadb_where_clause(
    cleaned: Dict[str, Any],
    max_filter_fields: int = 2,
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Konversi dict field yang sudah divalidasi → ChromaDB where clause.

    Operator per field:
        tahun  : "$eq"       — integer exact match (paling reliable)
        prodi  : "$contains" — substring match (toleran "Informatika" vs "S1 Informatika")
        penulis: "$contains" — substring match (toleran "Budi" vs "Budi Santoso")

    Return:
        (where_clause, applied_fields)

        where_clause   : dict ChromaDB siap pakai, atau None jika tidak ada filter valid
        applied_fields : list field yang BENAR-BENAR masuk ke where clause
                        setelah dipotong oleh max_filter_fields

    Jumlah kondisi dibatasi oleh max_filter_fields (default 2).
    """
    candidates: List[Tuple[str, Dict[str, Any]]] = []

    # Tambahkan kondisi berdasarkan field yang valid
    # Urutan: tahun dulu (paling reliable), lalu prodi, lalu penulis
    if cleaned.get("tahun") is not None:
        candidates.append(("tahun", {"tahun": {"$eq": int(cleaned["tahun"])}}))

    if cleaned.get("prodi"):
        candidates.append(("prodi", {"prodi": {"$contains": str(cleaned["prodi"])}}))

    if cleaned.get("penulis"):
        candidates.append(("penulis", {"penulis": {"$contains": str(cleaned["penulis"])}}))

    # Terapkan batas maksimum field filter
    selected = candidates[: int(max_filter_fields)]
    applied_fields = [field_name for field_name, _ in selected]
    conditions = [clause for _, clause in selected]

    if not conditions:
        return None, []

    if len(conditions) == 1:
        return conditions[0], applied_fields

    # 2+ kondisi: gabungkan dengan $and
    return {"$and": conditions}, applied_fields


# ═════════════════════════════════════════════════════════════════════════════
# Helper: bangun fallback result
# ═════════════════════════════════════════════════════════════════════════════

def _make_fallback(
    query: str,
    reason: str,
    llm_raw: Optional[str] = None,
    extracted: Optional[Dict[str, Any]] = None,
) -> SelfQueryResult:
    """
    Bangun SelfQueryResult dengan filters=None — caller wajib pakai retrieval penuh.
    Dipakai di setiap titik kegagalan dalam build_self_query().
    """
    return SelfQueryResult(
        semantic_query=query,           # selalu pakai query asli sebagai fallback
        filters=None,
        extracted_metadata=extracted or {},
        filter_applied=False,
        fallback_reason=reason,
        llm_raw_output=llm_raw,
        filter_fields_used=[],
    )


# ═════════════════════════════════════════════════════════════════════════════
# Public API — build_self_query
# ═════════════════════════════════════════════════════════════════════════════

def build_self_query(cfg: Dict[str, Any], query: str) -> SelfQueryResult:
    """
    Core function V3.0b: query bahasa alami → SelfQueryResult.

    Alur internal:
        1. Cek self_query.enabled dari config
        2. Build LLM dengan temperature=0.0
        3. Invoke LLM dengan few-shot prompt
        4. Ekstrak JSON dari output LLM (toleran terhadap teks tambahan)
        5. Parse + validasi field JSON
        6. Bangun ChromaDB where clause
        7. Return SelfQueryResult

    CATATAN PENTING:
        Fungsi ini TIDAK PERNAH melempar exception ke caller.
        Setiap titik kegagalan menghasilkan fallback_result dengan filters=None.
        Caller (app_streamlit.py) cukup cek .filter_applied untuk memutuskan
        apakah filter dipakai atau fallback ke retrieval penuh.

    Args:
        cfg   : config dict dari load_config()
        query : user query (bahasa alami Indonesia)

    Returns:
        SelfQueryResult — selalu valid, tidak pernah None
    """
    query = (query or "").strip()
    if not query:
        return _make_fallback(query, "empty_query")

    # ── Langkah 1: cek apakah fitur diaktifkan ─────────────────────────────────
    sq_cfg      = cfg.get("self_query") or {}
    enabled     = bool(sq_cfg.get("enabled", False))
    if not enabled:
        return _make_fallback(query, "self_query_disabled")

    raw_max_filter_fields = sq_cfg.get("max_filter_fields", 2)
    try:
        max_filter_fields = int(raw_max_filter_fields)
    except (TypeError, ValueError):
        max_filter_fields = 2

    # Sanitasi agar aman:
    # - minimal 1 field
    # - maksimal sebanyak field yang memang didukung
    max_filter_fields = max(1, min(max_filter_fields, len(_FILTER_FIELDS)))

    # ── Langkah 2: build LLM ───────────────────────────────────────────────────
    try:
        llm = _build_llm_for_self_query(cfg)
    except Exception as exc:
        return _make_fallback(
            query,
            f"llm_build_failed:{type(exc).__name__}:{exc}",
        )

    # ── Langkah 3: invoke LLM dengan few-shot prompt ───────────────────────────
    few_shot_text = _build_few_shot_text()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Kamu adalah asisten ekstraksi metadata dari query pencarian dokumen skripsi.\n"
                "Tugas: Dari query bahasa Indonesia, ekstrak metadata eksplisit (jika ada)\n"
                "dan pisahkan bagian topik/konten ke dalam semantic_query.\n\n"
                "Field metadata yang bisa diekstrak:\n"
                "  tahun  : tahun terbit skripsi (integer 2000-2030), null jika tidak disebut\n"
                "  prodi  : nama program studi/jurusan, null jika tidak disebut\n"
                "  penulis: nama penulis skripsi (bukan NIM), null jika tidak disebut\n\n"
                "ATURAN WAJIB:\n"
                "1. Output HANYA JSON — tidak ada penjelasan, tidak ada markdown fence.\n"
                # Kurung kurawal di-escape dengan {{ dan }}
                '2. Format: {{"semantic_query": "...", "tahun": <int|null>, '
                '"prodi": <str|null>, "penulis": <str|null>}}\n'
                "3. semantic_query = bagian TOPIK query tanpa info metadata.\n"
                "4. Jika query tidak menyebut metadata, isi tahun/prodi/penulis = null.\n"
                "5. tahun HARUS integer (bukan string). Contoh: 2023 bukan '2023'.\n"
                "6. JANGAN menebak metadata yang tidak disebut eksplisit.\n\n"
                "CONTOH:\n"
                f"{few_shot_text}",
            ),
            (
                "human",
                'Query: "{query}"\nOutput:',
            ),
        ]
    )

    chain   = prompt | llm | StrOutputParser()
    llm_raw: Optional[str] = None

    try:
        llm_raw = chain.invoke({"query": query}).strip()
    except Exception as exc:
        return _make_fallback(
            query,
            f"llm_call_failed:{type(exc).__name__}:{exc}",
            llm_raw=None,
        )

    # ── Langkah 4: ekstrak JSON dari output LLM ────────────────────────────────
    json_str = _extract_json_from_llm_output(llm_raw)
    if not json_str:
        return _make_fallback(
            query,
            "json_extraction_failed:empty_output",
            llm_raw=llm_raw,
        )

    try:
        raw_dict: Dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return _make_fallback(
            query,
            f"json_parse_failed:{exc}",
            llm_raw=llm_raw,
        )

    if not isinstance(raw_dict, dict):
        return _make_fallback(
            query,
            "json_not_a_dict",
            llm_raw=llm_raw,
        )

    # ── Langkah 5: validasi & bersihkan field ──────────────────────────────────
    try:
        cleaned = _validate_and_clean_extracted(raw_dict)
    except Exception as exc:
        return _make_fallback(
            query,
            f"validation_failed:{type(exc).__name__}:{exc}",
            llm_raw=llm_raw,
            extracted=raw_dict,
        )

    # semantic_query dari LLM; fallback ke query asli jika kosong
    semantic_query = cleaned.get("semantic_query") or query

    # ── Langkah 6: bangun ChromaDB where clause ────────────────────────────────
    try:
        where_clause, fields_used = _build_chromadb_where_clause(cleaned, max_filter_fields)
    except Exception as exc:
        return _make_fallback(
            query,
            f"where_clause_build_failed:{type(exc).__name__}:{exc}",
            llm_raw=llm_raw,
            extracted=cleaned,
        )

    # Jika tidak ada filter valid → fallback
    # (ini kasus normal: query umum tanpa metadata intent)
    if not fields_used or where_clause is None:
        return _make_fallback(
            query,
            "no_valid_filters_extracted",
            llm_raw=llm_raw,
            extracted=cleaned,
        )

    # ── Langkah 7: return hasil ────────────────────────────────────────────────
    return SelfQueryResult(
        semantic_query=semantic_query,
        filters=where_clause,
        extracted_metadata=cleaned,
        filter_applied=True,
        fallback_reason=None,
        llm_raw_output=llm_raw,
        filter_fields_used=fields_used,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Public API — check_filter_has_results (graceful fallback pre-check)
# ═════════════════════════════════════════════════════════════════════════════

def check_filter_has_results(
    where_filter: Dict[str, Any],
    cfg: Dict[str, Any],
    collection: str = "narasi",
) -> bool:
    """
    Pre-check: apakah filter menghasilkan setidaknya 1 dokumen di ChromaDB.

    INI adalah implementasi utama dari GRACEFUL FALLBACK wajib V3.0b:
    "Jika ChromaDB where clause filter mengembalikan 0 hasil (karena metadata
    tidak terdeteksi saat ingest), sistem WAJIB fallback ke retrieval penuh."

    Dipanggil oleh app_streamlit.py SEBELUM mengirim filter ke retrieve_dense().
    Jika return False → caller langsung jalankan retrieval penuh tanpa filter.

    Kenapa perlu pre-check terpisah (bukan langsung di retrieve_dense)?
    Karena retrieve_dense() akan crash/return empty jika filter tidak cocok
    dengan dokumen manapun, lebih baik deteksi lebih awal sebelum retrieval.

    Args:
        where_filter : ChromaDB where clause dict dari SelfQueryResult.filters
        cfg          : config dict dari load_config()
        collection   : collection yang akan di-query (default "narasi")

    Returns:
        True  : ada ≥1 dokumen yang cocok → aman untuk apply filter
        False : 0 dokumen cocok atau terjadi error → wajib fallback ke full retrieval
    """
    if not where_filter:
        return False

    try:
        # Import di dalam fungsi untuk mencegah circular import
        # (self_query.py ← app_streamlit.py ← retrieve_dense.py)
        from src.rag.retrieve_dense import _get_vectorstore  # type: ignore

        vs         = _get_vectorstore(cfg, collection)
        chroma_col = vs._collection  # underlying chromadb.Collection

        # Gunakan .get() dengan where clause dan limit=1 untuk efisiensi
        # "ids" selalu dikembalikan otomatis (tidak perlu di include list)
        result = chroma_col.get(
            where=where_filter,
            limit=1,
            include=[],
        )
        ids = result.get("ids") or []
        return len(ids) > 0

    except Exception:
        # Jika pre-check gagal karena alasan apapun (ChromaDB error,
        # operator tidak didukung, dll.) → aman untuk fallback ke full retrieval
        return False