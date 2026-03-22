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
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

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