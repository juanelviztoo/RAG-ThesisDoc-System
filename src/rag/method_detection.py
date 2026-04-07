"""
method_detection.py
===================
Shared utilities untuk deteksi metode & query type.
Digunakan di generate_utils.py dan app_streamlit.py.

Menggantikan duplikasi antara kedua file:
    - _METHOD_LABEL (generate) / _METHOD_PRETTY (app)            → METHOD_LABELS
    - _METHOD_MATCH_PATTERNS (generate) / _METHOD_PATTERNS (app) → METHOD_PATTERNS (superset)
    - _nice_method() / pretty_method_name()                      → pretty_method_name()
    - _has_steps_signal_local() / has_steps_signal()             → has_steps_signal() [master]
    - detect_methods_in_text/contexts()                          → pindah dari app_streamlit
    - is_*_question()                                            → pindah dari app_streamlit
    - _docid_hit_for_method()                                    → docid_hit_for_method()
    - node_supports_method_for_coverage()                        → pindah dari app_streamlit

V3.0b (Konsolidasi):
    - is_citation_query() — single source of truth untuk deteksi citation query.
        Menggantikan duplikasi:
        * _is_citation_query() + _CITATION_KEYWORDS di app_streamlit.py (dihapus)
        * referensi ke is_citation_query() di pdf_parser.py (tidak perlu ada lagi)
        Semua caller cukup import dari modul ini.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# =========================
# Konstanta — single source of truth
# =========================

# Menggantikan _METHOD_PRETTY (app_streamlit) dan _METHOD_LABEL (generate).
METHOD_LABELS: Dict[str, str] = {
    "RAD": "RAD",
    "PROTOTYPING": "Prototyping",
    "RUP": "RUP",
    "EXTREME_PROGRAMMING": "Extreme Programming",
    "WATERFALL": "Waterfall",
    "AGILE": "Agile",
}

# Superset dari _METHOD_PATTERNS (app_streamlit) dan _METHOD_MATCH_PATTERNS (generate).
# Pola yang lebih broad dipertahankan; pola duplikat dibuang.
METHOD_PATTERNS: Dict[str, List[str]] = {
    "RAD": [
        r"\brad\b",
        r"rapid application development",
        r"rapidapplicationdevelopment",
    ],
    "PROTOTYPING": [
        r"\bprototyp\w*\b",
        r"\bprototype\b",
        r"\bmodel\s+prototyp\w*\b",
        r"\bmetode\s+prototyp\w*\b",
    ],
    "RUP": [
        r"\brup\b",
        r"rational unified process",
        r"rationalunifiedprocess",
    ],
    "EXTREME_PROGRAMMING": [
        r"extreme\s*programming",
        r"extreme[_\-\s]*programming",
        r"extremeprogramming",
        r"\bxp\b",
    ],
    "WATERFALL": [r"\bwaterfall\b"],
    "AGILE": [r"\bagile\b", r"\bscrum\b"],
}

# Dipakai untuk matching doc_id ke metode (coverage retrieval).
COVERAGE_DOCID_HINTS: Dict[str, List[str]] = {
    "PROTOTYPING": ["prototyp"],
    "RAD": ["_rad", "rad"],
    "EXTREME_PROGRAMMING": ["extremeprogramming", "extreme_programming", "xp"],
    "RUP": ["_rup", "rup"],
    "WATERFALL": ["waterfall"],
    "AGILE": ["agile", "scrum"],
}

# Pola teks ketat untuk coverage node-matching (khususnya PROTOTYPING agar tidak false-positive).
COVERAGE_STRICT_TEXT_PATTERNS: Dict[str, List[str]] = {
    "PROTOTYPING": [
        r"\bprototyping\b",
        r"\bmetode\s+prototyp",
        r"\bmodel\s+prototyp",
        r"\bpendekatan\s+prototyp",
    ],
    "RAD": [
        r"\brad\b",
        r"rapid application development",
        r"\bmetode\s+rad\b",
    ],
}


# =========================
# Citation query keywords — V3.0b
# (Sebelumnya: _CITATION_KEYWORDS di app_streamlit.py - dihapus dari sana)
# =========================

# Superset dari kedua implementasi lama. Setiap perubahan keyword cukup
# dilakukan di sini - app_streamlit.py dan modul lain cukup import is_citation_query().
_CITATION_KEYWORDS: List[str] = [
    # ── Intent eksplisit: menyebut daftar pustaka / referensi secara langsung ──
    r"\b(referensi|daftar\s*pustaka|daftar\s*referensi)\b",
    # ── Intent implisit: menyebut jenis sumber + kata kerja/pertanyaan ──
    r"\b(jurnal|buku|artikel|publikasi|acuan|sitasi)\b.*\b(digunakan|dijadikan|ada|apa)\b",
    r"\b(apa|sebutkan|tampilkan|cari)\b.*\b(jurnal|buku|referensi|pustaka|acuan)\b",
    # ── Frasa akademis yang sangat khas bagian literatur ──
    r"\b(landasan\s*teori|tinjauan\s*pustaka|kajian\s*literatur)\b",
    # ── Kata kerja turunan sumber: mengutip, sumber pustaka ──
    r"\b(mengutip|dikutip|menggunakan\s+referensi|sumber\s+pustaka)\b",
    # ── Bahasa Inggris (dokumen bilingual / query campur) ──
    r"\b(cited|citation|bibliography|references?)\b",
]


# =========================
# V3.0d lanjutan — Intent planning
# =========================

@dataclass
class IntentPlan:
    """
    Planner output untuk answer policy V3.0d lanjutan.

    Catatan desain:
    - primary_intent = route utama generator/runtime
    - requested_actions = sub-intent / aksi yang harus dipertahankan
    - target_methods = metode yang secara eksplisit terdeteksi dari query/history
    - comparison_targets = alias yang lebih semantik untuk comparison route
    """

    primary_intent: str = "general_qa"
    requested_actions: List[str] = field(default_factory=list)

    target_methods: List[str] = field(default_factory=list)
    comparison_targets: List[str] = field(default_factory=list)

    doc_locked: bool = False
    allowed_doc_ids: List[str] = field(default_factory=list)

    needs_doc_aggregation: bool = False
    needs_explanation_expansion: bool = False
    needs_steps_expansion: bool = False

    is_multi_doc: bool = False
    is_compound: bool = False
    is_count_query: bool = False
    is_list_query: bool = False
    is_explanation_query: bool = False
    is_steps_query: bool = False
    is_comparison_query: bool = False

    requested_fields: List[str] = field(default_factory=list)
    planner_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _query_has_any(text: str, patterns: Sequence[str]) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t, flags=re.IGNORECASE) for p in (patterns or []))


def is_method_identification_question(q: str) -> bool:
    """
    True bila user terutama meminta IDENTIFIKASI nama metode,
    bukan penjelasan atau tahapan.
    """
    ql = (q or "").lower()
    if not is_method_question(ql):
        return False
    if is_steps_question(ql):
        return False
    if is_method_explanation_question(ql):
        return False
    return bool(
        re.search(
            r"\b(metode|model)\b.*\b(apa|apakah)\b|\bapa\b.*\b(metode|model)\b|\bmenggunakan\b.*\bmetode\s+apa\b",
            ql,
            flags=re.IGNORECASE,
        )
    )


def is_method_explanation_question(q: str) -> bool:
    """
    True bila user meminta PENJELASAN metode dalam konteks dokumen,
    bukan sekadar nama metodenya.
    """
    ql = (q or "").lower()
    explain_patterns = [
        r"\bjelaskan\b",
        r"\bpenjelasan\b",
        r"\bdigunakan\s+seperti\s+apa\b",
        r"\bbagaimana\b[^\n]{0,40}\bdigunakan\b",
        r"\balasan\b[^\n]{0,20}\bmenggunakan\b",
        r"\bkarakteristik\b",
        r"\bkonteks\b[^\n]{0,20}\bpenggunaan\b",
        r"\bterkait\b",
    ]
    if not _query_has_any(ql, explain_patterns):
        return False

    # Query count/list yang kebetulan mengandung
    # "Jika ada, jelaskan ..." tetap ditangani sebagai compound intent di planner utama.
    return True


def is_count_query(q: str) -> bool:
    ql = (q or "").lower()
    return bool(
        re.search(
            r"\b(ada\s+berapa|berapa\s+(skripsi|dokumen|file)|jumlah\s+(skripsi|dokumen|file)|adakah)\b",
            ql,
            flags=re.IGNORECASE,
        )
    )


def is_listing_query(q: str) -> bool:
    ql = (q or "").lower()
    return bool(
        re.search(
            r"\b(sebutkan|daftarkan|tampilkan|skripsi\s+yang\s+menggunakan|dokumen\s+yang\s+menggunakan|judul(?:nya)?|penulis(?:nya)?)\b",
            ql,
            flags=re.IGNORECASE,
        )
    )


def is_comparison_question(q: str) -> bool:
    ql = (q or "").lower()
    if re.search(r"\b(perbedaan|perbandingan|bandingkan|dibandingkan)\b", ql, flags=re.IGNORECASE):
        return True
    methods = detect_methods_in_text(ql)
    return len(methods) >= 2 and bool(
        re.search(r"\b(vs|versus|dan|&|antara)\b", ql, flags=re.IGNORECASE)
    )


def _prefer_method_explanation_over_listing(
    *,
    is_explanation: bool,
    is_count: bool,
    is_steps: bool,
    is_comparison: bool,
    requested_fields: Sequence[str],
    methods: Sequence[str],
) -> bool:
    """
    True jika query lebih tepat diperlakukan sebagai single-method explanation daripada doc listing.

    Kasus target:
    - "Pada skripsi yang menggunakan metode RAD, tolong jelaskan RAD yang digunakan di sana!"
    - "Pada dokumen yang memakai XP, jelaskan XP yang dipakai di sana!"

    Prinsip:
    - explanation tunggal harus menang atas listing semu yang hanya muncul
        karena frasa seperti 'skripsi yang menggunakan ...'
    - tetapi TIDAK boleh mengambil alih query count/list yang memang eksplisit
    """
    if not is_explanation:
        return False
    if is_count or is_steps or is_comparison:
        return False
    if requested_fields:
        return False

    method_count = len([m for m in (methods or []) if m])
    return method_count <= 1


def build_intent_plan(
    query: str,
    *,
    history_window: Optional[Sequence[Dict[str, Any]]] = None,
    doc_focus: Optional[Sequence[str]] = None,
) -> IntentPlan:
    """
    Intent planner sederhana namun eksplisit untuk Tahap 1.

    Scope Tahap 1:
    - menentukan route utama yang benar
    - menjaga compound intent agar tidak hilang
    - belum melakukan doc aggregation / explanation expansion itu sendiri
    """
    q = (query or "").strip()
    ql = q.lower()

    methods = detect_methods_in_text(q)
    if (not methods) and history_window:
        # fallback ringan untuk anaphora/steps
        hist_blob = " ".join(
            f"{h.get('query', '')} {h.get('answer_preview', '')}"
            for h in (history_window or [])
            if isinstance(h, dict)
        )
        if is_anaphora_question(q) or is_steps_question(q):
            methods = detect_methods_in_text(hist_blob)

    methods = list(dict.fromkeys([m for m in methods if m]))

    is_steps = is_steps_question(q)
    is_comparison = is_comparison_question(q)
    is_count = is_count_query(q)
    is_list = is_listing_query(q)
    is_explanation = is_method_explanation_question(q)
    is_multi_doc = is_multi_doc_question(q) or bool(
        re.search(
            r"\b(beberapa\s+skripsi|di\s+beberapa\s+skripsi|dokumen\s+skripsi\s+yang\s+ada|lintas\s+dokumen)\b",
            ql,
            flags=re.IGNORECASE,
        )
    )

    requested_fields: List[str] = []
    if re.search(r"\bjudul(?:nya)?\b", ql, flags=re.IGNORECASE):
        requested_fields.append("title")
    if re.search(r"\bpenulis(?:nya)?\b", ql, flags=re.IGNORECASE):
        requested_fields.append("author")

    actions: List[str] = ["show_evidence"]
    reasons: List[str] = []

    if is_count:
        actions.append("count_docs")
        reasons.append("count_signal")
    if is_list or requested_fields:
        actions.append("list_docs")
        reasons.append("list_signal")
    if is_explanation:
        actions.append("explain_method")
        reasons.append("explanation_signal")
    if is_steps:
        actions.append("extract_steps")
        reasons.append("steps_signal")
    if is_comparison:
        actions.append("compare_methods")
        reasons.append("comparison_signal")
    if re.search(r"\b(metode|model)\b", ql, flags=re.IGNORECASE):
        actions.append("identify_method")

    if "title" in requested_fields:
        actions.append("show_title")
    if "author" in requested_fields:
        actions.append("show_author")

    actions = list(dict.fromkeys(actions))

    prefer_explanation_over_listing = _prefer_method_explanation_over_listing(
        is_explanation=is_explanation,
        is_count=is_count,
        is_steps=is_steps,
        is_comparison=is_comparison,
        requested_fields=requested_fields,
        methods=methods,
    )

    if prefer_explanation_over_listing:
        reasons.append("explanation_preempts_listing")

    primary_intent = "general_qa"
    if is_comparison:
        primary_intent = "method_comparison"
    elif is_steps and len(methods) <= 1:
        primary_intent = "single_target_steps"
    elif is_count and (is_list or requested_fields or is_multi_doc or is_explanation):
        primary_intent = "doc_count_list"
    elif prefer_explanation_over_listing:
        primary_intent = "method_explanation"
    elif is_list and (is_multi_doc or bool(methods)):
        primary_intent = "doc_listing"
    elif is_explanation and bool(methods or is_method_question(q)):
        primary_intent = "method_explanation"
    elif is_method_identification_question(q):
        primary_intent = "method_identification"

    allowed_doc_ids = [str(x).strip() for x in (doc_focus or []) if str(x).strip()]
    allowed_doc_ids = list(dict.fromkeys(allowed_doc_ids))

    needs_doc_aggregation = primary_intent in {"doc_count_list", "doc_listing"}
    if primary_intent == "method_explanation" and is_multi_doc:
        needs_doc_aggregation = True

    # Tahap 4:
    # single_target_steps yang BELUM doc-locked perlu doc aggregation agar runtime bisa:
    # - resolve owner doc tunggal, atau
    # - gagal aman sebagai ambiguous_owner_docs
    if primary_intent == "single_target_steps" and (not allowed_doc_ids):
        needs_doc_aggregation = True

    return IntentPlan(
        primary_intent=primary_intent,
        requested_actions=actions,
        target_methods=methods,
        comparison_targets=(methods[:] if is_comparison else []),
        doc_locked=bool(allowed_doc_ids),
        allowed_doc_ids=allowed_doc_ids,
        needs_doc_aggregation=bool(needs_doc_aggregation),
        needs_explanation_expansion=bool(primary_intent == "method_explanation"),
        needs_steps_expansion=bool(primary_intent == "single_target_steps"),
        is_multi_doc=bool(is_multi_doc),
        is_compound=bool(len(actions) > 2),
        is_count_query=bool(is_count),
        is_list_query=bool(is_list),
        is_explanation_query=bool(is_explanation),
        is_steps_query=bool(is_steps),
        is_comparison_query=bool(is_comparison),
        requested_fields=requested_fields,
        planner_reason="; ".join(reasons) if reasons else "fallback_general",
    )


# =========================
# Pretty-print helper
# =========================

def pretty_method_name(m: str) -> str:
    """
    Return display label untuk method key.
    Menggantikan _nice_method() di generate.py dan pretty_method_name() di app_streamlit.py.
    Contoh: 'EXTREME_PROGRAMMING' → 'Extreme Programming'.
    """
    return METHOD_LABELS.get(m, m)


# =========================
# Method detection
# =========================

def _normalize_for_method_detection(text: str) -> str:
    """
    Normalisasi agar deteksi metode robust untuk:
    - doc_id dengan underscore: ITERA_..._Prototyping
    - CamelCase: ExtremeProgramming
    """
    s = text or ""
    # pecah CamelCase: ExtremeProgramming -> Extreme Programming
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    # underscore/hyphen sebagai separator
    s = re.sub(r"[_\-]+", " ", s)
    return " ".join(s.split()).lower()


def detect_methods_in_text(text: str) -> List[str]:
    """
    Ambil daftar metode terdeteksi dari sebuah teks (case-insensitive, deduplicated).
    Normalisasi underscore & CamelCase sudah dihandle.
    """
    t = _normalize_for_method_detection(text)
    found: List[str] = []
    for name, pats in METHOD_PATTERNS.items():
        for p in pats:
            if re.search(p, t, flags=re.IGNORECASE):
                found.append(name)
                break

    # unik + urut stabil
    out: List[str] = []
    seen: set = set()
    for x in found:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def detect_methods_in_contexts(contexts: List[str]) -> List[str]:
    """Deteksi metode dari gabungan list context strings."""
    return detect_methods_in_text("\n".join(contexts or []))


# =========================
# Steps signal — master version (single source of truth)
#
# Menggantikan DUA implementasi yang berbeda:
#   - has_steps_signal()       di app_streamlit.py  (versi lebih lengkap, pakai connector logic)
#   - _has_steps_signal_local() di generate.py       (versi lebih sederhana, tanpa [a-c] & connector)
#
# Versi master = versi app_streamlit.py (lebih lengkap).
# =========================

def has_steps_signal(text: str) -> bool:
    """
    Sinyal bahwa sebuah chunk/ctx benar-benar berisi info tahapan/langkah.
    Return True jika chunk memuat sinyal tahapan yang eksplisit.

    Sinyal kuat (diurutkan dari paling kuat):
    1. Keyword 'tahap/tahapan/langkah/prosedur/fase'
    2. Enumerasi eksplisit (1., 2), a), -, •)
    3. Connector daftar ('meliputi/terdiri/yaitu') + >= 2 stage terms

    Konservatif: kata generik seperti 'iterasi/feedback' TIDAK trigger
    tanpa adanya connector eksplisit.
    """
    t = (text or "").lower()
    # buang header "DOC: ..." agar tidak mengganggu sinyal
    t = re.sub(r"(?im)^doc:\s.*\n", "", t).strip()

    # sinyal kuat: keyword eksplisit (kata "tahap/tahapan/langkah/prosedur/fase")
    if re.search(r"\b(tahap|tahapan|langkah|prosedur|fase)\w*\b", t):
        return True

    # sinyal kuat: enumerasi
    #    contoh: "1.", "2)", "a)", "b.", "- " (untuk daftar yang jelas)
    if re.search(r"(?m)^\s*(\d+[\.\)]|[a-c][\.\)]|[-•])\s+", t):
        return True

    # sinyal sedang: connector yang biasanya memperkenalkan daftar tahapan
    has_connector = bool(
        re.search(r"\b(meliputi|terdiri(\s+dari)?|yaitu|antara\s+lain)\b", t)
    )

    # sinyal sedang: stage terms, hanya dihitung jika ada connector + minimal 2 stage terms
    stage_terms = [
        "planning", "perencanaan", "design", "perancangan",
        "implementation", "implementasi", "coding", "pengkodean",
        "testing", "pengujian", "evaluasi", "deployment",
        "pemeliharaan", "analisis", "desain",
    ]
    stage_hits = sum(1 for s in stage_terms if re.search(rf"\b{re.escape(s)}\b", t))

    if has_connector and stage_hits >= 2:
        return True

    # kalau tidak ada sinyal eksplisit di atas, anggap tidak memuat tahapan
    return False


# =========================
# Query type detection
# (pindah dari app_streamlit.py — dipakai oleh app dan generate_utils)
# =========================

def is_method_question(q: str) -> bool:
    """True jika query menanyakan metode/model pengembangan."""
    return bool(re.search(r"\b(metode|model)\b", (q or "").lower()))


def is_steps_question(q: str) -> bool:
    """
    True jika query menanyakan tahapan/langkah/alur/prosedur dalam konteks
    pengembangan sistem/metode.

    Perketat deteksi untuk kata 'alur' dan 'prosedur':
    - 'tahap/tahapan/langkah' → selalu trigger (kata teknis yang spesifik)
    - 'alur/prosedur' → hanya trigger jika ada konteks teknis
        (mencegah false positive: "alur penelitian", "alur metodologi penelitian")
    """
    ql = (q or "").lower()

    # "tahap/tahapan/langkah" - kata teknis spesifik, selalu trigger
    if re.search(r"\b(tahap|tahapan)\w*\b|\blangkah\w*\b", ql):
        return True

    # "alur/prosedur" - hanya trigger jika ada konteks teknis pengembangan sistem
    _TECH_CONTEXT = r"\b(sistem|pengembangan|aplikasi|software|perangkat\s*lunak|metode\s+pengembangan|sdlc)\b"
    if re.search(r"\b(alur|prosedur)\w*\b", ql):
        return bool(re.search(_TECH_CONTEXT, ql))

    return False


def is_anaphora_question(q: str) -> bool:
    """
    Deteksi pertanyaan lanjutan yang biasanya butuh konteks sebelumnya.
    True jika query memakai referensi anafora ('itu', 'tersebut', 'nya', dll.).
    """
    return bool(
        re.search(
            r"\b(itu|tersebut|ini|nya|tahapannya|langkahnya|alur\s*nya|metodenya)\b",
            (q or "").lower(),
        )
    )


def is_multi_target_question(q: str) -> bool:
    """
    Deteksi pertanyaan yang meminta jawaban mencakup beberapa target/metode.
    Contoh: 'semua metode', 'masing-masing metode', 'RAD dan prototyping', dll.
    True jika query meminta jawaban untuk beberapa metode sekaligus.
    """
    t = (q or "").lower()
    # frasa eksplisit
    key_phrases = [
        "semua metode", "masing-masing", "setiap metode", "beberapa metode",
        "dua metode", "kedua metode", "metode rad dan", "rad dan prototyping",
        "prototyping dan rad", "rad & prototyping", "metode apa saja",
        "metode-metode", "masing masing",
    ]
    if any(p in t for p in key_phrases):
        return True
    # pola regex: (semua/masing-masing/setiap) + (metode/model)
    if re.search(r"\b(semua|masing-masing|masing|setiap|beberapa)\b.*\b(metode|model)\b", t):
        return True
    # ada beberapa metode disambung "dan"
    if re.search(r"\b(rad|rapid application development)\b.*\b(dan|&)\b.*\b(proto|prototyp)", t):
        return True
    return False


def is_multi_doc_question(q: str) -> bool:
    """True jika query menanyakan lintas beberapa dokumen/sumber."""
    t = (q or "").lower()
    # frasa eksplisit
    key_phrases = [
        "beberapa dokumen", "dokumen lain", "dokumen lainnya", "berbagai dokumen",
        "lebih dari satu dokumen", "lintas dokumen", "multi dokumen",
        "beberapa sumber", "sumber lain", "di dokumen berbeda",
    ]
    if any(p in t for p in key_phrases):
        return True
    # pola umum: ada kata "beberapa/berbagai" dan "dokumen/sumber"
    if re.search(r"\b(beberapa|berbagai|multi)\b.*\b(dokumen|sumber)\b", t):
        return True
    return False


def is_citation_query(query: str) -> bool:
    """
    Deteksi apakah query meminta informasi dari Daftar Pustaka / referensi.
    Jika True → pipeline akan menambahkan retrieval dari collection sitasi.

    Single source of truth - menggantikan duplikasi:
        - _is_citation_query() + _CITATION_KEYWORDS di app_streamlit.py (sudah dihapus)
        - referensi is_citation_query() di pdf_parser.py (tidak perlu ada lagi)

    Digunakan oleh:
        - app_streamlit.py  : citation routing (hybrid + dense path)
        - (future) self_query.py : sebagai sinyal bahwa filter sitasi perlu diaktifkan
    """
    q = (query or "").lower()
    return any(re.search(p, q, flags=re.IGNORECASE) for p in _CITATION_KEYWORDS)


# =========================
# Coverage helpers
# (pindah dari app_streamlit.py — docid + node coverage check)
# =========================

def docid_hit_for_method(doc_id: str, method: str) -> bool:
    """
    True jika doc_id mengandung hint untuk metode yang dimaksud.
    Menggantikan _docid_hit_for_method() di app_streamlit.py.
    """
    did = (doc_id or "").lower()
    for h in COVERAGE_DOCID_HINTS.get(method, []):
        if h and h in did:
            return True
    return False


def node_supports_method_for_coverage(n: Any, method: str) -> bool:
    """
    True jika node benar-benar mendukung 'method' untuk coverage retrieval.

    Khusus PROTOTYPING: HARUS ada doc_id match atau pola teks eksplisit.
    Tidak ada fallback generik agar tidak false-positive.
    """
    doc_id = getattr(n, "doc_id", "") or ""
    text = (getattr(n, "text", "") or "").lower()

    # 1) doc_id hint (paling stabil)
    if docid_hit_for_method(doc_id, method):
        return True

    # 2) strict text patterns
    for p in COVERAGE_STRICT_TEXT_PATTERNS.get(method, []):
        if p and re.search(p, text, flags=re.IGNORECASE):
            return True

    # 3) fallback HANYA untuk metode selain PROTOTYPING
    if method == "PROTOTYPING":
        return False

    return method in set(detect_methods_in_text(text))