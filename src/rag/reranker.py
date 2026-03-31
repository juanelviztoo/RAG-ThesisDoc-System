"""
reranker.py — Cross-Encoder Reranking Module (V3.0c)

Tanggung jawab:
    - Mengelola lifecycle Cross-Encoder: load → score → unload
    - Mengekspos interface bersih untuk app_streamlit.py
    - Sequential Loading (WAJIB): modul ini di-load SEBELUM LLM dimuat,
    dan di-unload SETELAH reranking selesai, SEBELUM LLM dimuat.

Kontrak penting:
    - rerank() selalu NON-MUTATING: input sequence tidak dimodifikasi sama sekali
    - rerank() selalu kembalikan list[RetrievedNode] BARU (via dataclasses.replace)
    - Jika model tidak loaded / load gagal:
        → kembalikan nodes[:top_n_rerank] tanpa scoring (graceful skip)
        → rerank_score=None, rerank_rank=None untuk semua node
    - Modul ini tidak boleh tahu tentang LLM, UI, atau pipeline retrieval

Aturan VRAM (RTX 3050 Ti 4GB):
    - Jangan load Cross-Encoder dan LLM secara bersamaan
    - Urutan wajib di app_streamlit.py:
        load_reranker(cfg)
        nodes = rerank(query, retrieved_nodes, top_n_rerank)
        unload_reranker()
        # ... load LLM di sini (via Groq/Ollama) ...
        answer = generate_answer(...)

    ⚠️ PERINGATAN INTEGRASI — JANGAN SORT ULANG SETELAH rerank():
        Urutan list yang dikembalikan rerank() adalah urutan final berdasarkan
        Cross-Encoder score. Jangan ada sort berdasarkan n.score (yang menyimpan
        skor retrieval lama: score_rrf untuk hybrid, distance untuk dense).
        Relevansi final ada di urutan list dan rerank_rank, BUKAN di n.score.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
from collections.abc import Sequence
from typing import Any, Dict, List, Optional

from src.core.schemas import RetrievedNode

logger = logging.getLogger(__name__)

# ── Module-level state ────────────────────────────────────────────────────────
_reranker_model: Any = None               # CrossEncoder instance atau None
_reranker_model_name: Optional[str] = None
_reranker_resolved_device: Optional[str] = None   # device yang benar-benar dipakai
_reranker_requested_device: Optional[str] = None  # device yang diminta config
_load_failed: bool = False                # True jika load() terakhir gagal
_load_error_msg: Optional[str] = None
_load_fallback_reason: Optional[str] = None       # alasan fallback device (jika ada)


# ── Device Resolution ─────────────────────────────────────────────────────────

def _resolve_device(cfg: Dict[str, Any]) -> tuple[str, str, Optional[str]]:
    """
    Tentukan device untuk Cross-Encoder berdasarkan config dan ketersediaan hardware.

    Returns:
        Tuple (requested_device, resolved_device, fallback_reason)
        - requested_device : nilai device dari config ("auto" | "cpu" | "cuda")
        - resolved_device  : device yang benar-benar akan dipakai ("cpu" | "cuda")
        - fallback_reason  : alasan fallback jika resolved != requested, atau None
    """
    reranker_cfg = cfg.get("reranker", {})
    device_pref = reranker_cfg.get("device", "auto").lower()
    min_vram_mb = int(reranker_cfg.get("min_vram_mb", 800))

    if device_pref == "cpu":
        return "cpu", "cpu", None

    if device_pref == "cuda":
        # Paksa CUDA — tidak ada fallback, resiko ada di caller
        return "cuda", "cuda", None

    # Auto-detect
    try:
        import torch

        if not torch.cuda.is_available():
            reason = "CUDA tidak tersedia di sistem"
            logger.info("[reranker] %s → pakai CPU", reason)
            return "auto", "cpu", reason

        free_vram_mb = _get_free_vram_mb()
        if free_vram_mb is not None and free_vram_mb < min_vram_mb:
            reason = f"Free VRAM {free_vram_mb:.0f} MB < threshold {min_vram_mb} MB"
            logger.warning("[reranker] %s → fallback CPU", reason)
            return "auto", "cpu", reason

        logger.info(
            "[reranker] CUDA tersedia, free VRAM ~%.0f MB → pakai CUDA",
            free_vram_mb or 0.0,
        )
        return "auto", "cuda", None

    except ImportError:
        reason = "PyTorch tidak terinstall"
        logger.info("[reranker] %s → pakai CPU", reason)
        return "auto", "cpu", reason


def _get_free_vram_mb() -> Optional[float]:
    """
    Kembalikan free VRAM dalam MB untuk GPU device 0.
    Kembalikan None jika tidak bisa dideteksi (tidak ada GPU / error).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, _ = torch.cuda.mem_get_info(0)
        return free_bytes / (1024 * 1024)
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def load_reranker(cfg: Dict[str, Any]) -> None:
    """
    Load Cross-Encoder ke memori.

    Behavior:
    - Jika model yang SAMA sudah loaded di device yang SAMA → skip (idempotent).
    - Jika model berbeda / device berbeda dan ada model lama → unload dulu, baru
        load baru. Ini mencegah memory retention diam-diam jika caller lupa unload.
    - Jika load gagal → set _load_failed=True, TIDAK raise exception.

    Caller harus cek is_reranker_loaded() atau get_reranker_info() untuk status.
    """
    global _reranker_model, _reranker_model_name, _reranker_resolved_device
    global _reranker_requested_device, _load_failed, _load_error_msg, _load_fallback_reason

    reranker_cfg = cfg.get("reranker", {})
    model_name = reranker_cfg.get("model_name", "BAAI/bge-reranker-base")

    # Resolve device terlebih dahulu untuk cek apakah perlu reload
    requested, resolved, fallback_reason = _resolve_device(cfg)

    # Jika model yang SAMA sudah loaded di device yang SAMA → skip
    if (
        _reranker_model is not None
        and _reranker_model_name == model_name
        and _reranker_resolved_device == resolved
    ):
        logger.debug(
            "[reranker] Model '%s' sudah loaded di device='%s', skip re-load",
            model_name,
            resolved,
        )
        return

    # Jika ada model lain yang sedang loaded (beda model atau beda device) → unload dulu
    if _reranker_model is not None:
        logger.info(
            "[reranker] Model lama '%s' (device=%s) akan di-unload sebelum load model baru",
            _reranker_model_name,
            _reranker_resolved_device,
        )
        unload_reranker()

    # Reset state sebelum load
    _load_failed = False
    _load_error_msg = None
    _load_fallback_reason = fallback_reason
    _reranker_requested_device = requested

    logger.info("[reranker] Loading '%s' ke device='%s'...", model_name, resolved)

    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        model = CrossEncoder(model_name, device=resolved)

        _reranker_model = model
        _reranker_model_name = model_name
        _reranker_resolved_device = resolved
        logger.info(
            "[reranker] Model '%s' loaded sukses "
            "(requested=%s, resolved=%s, fallback_reason=%s)",
            model_name,
            requested,
            resolved,
            fallback_reason or "none",
        )

    except Exception as exc:
        _reranker_model = None
        _reranker_model_name = None
        _reranker_resolved_device = None
        _load_failed = True
        _load_error_msg = str(exc)
        logger.error(
            "[reranker] Gagal load model '%s': %s", model_name, exc, exc_info=True
        )


def rerank(
    query: str,
    nodes: Sequence[RetrievedNode],
    top_n_rerank: int,
) -> list[RetrievedNode]:
    """
    Rerank nodes menggunakan Cross-Encoder (deep query-document interaction).

    Semua nodes yang dikirim masuk akan di-score. Kontrol jumlah kandidat
    (candidate_k) dilakukan oleh caller (app_streamlit.py) sebelum memanggil
    fungsi ini — bukan di sini.

    Graceful skip (jika model tidak loaded / load gagal):
        → kembalikan nodes[:top_n_rerank] tanpa scoring
        → rerank_score=None, rerank_rank=None untuk semua node

    Args:
        query        : Query string (effective_retrieval_query pasca-contextualization)
        nodes        : Kandidat hasil retrieval yang akan di-rerank
        top_n_rerank : Jumlah node final yang dikembalikan (minimum 1)

    Returns:
        list[RetrievedNode] BARU dengan rerank_score dan rerank_rank terisi.
        Urutan list adalah urutan FINAL — JANGAN sort ulang berdasarkan n.score.

    ⚠️ PERINGATAN: Jangan sort ulang output fungsi ini berdasarkan n.score.
        n.score menyimpan skor retrieval lama (score_rrf / distance).
        Relevansi final ada di urutan list ini dan field rerank_rank.
    """
    # Guard: pastikan top_n_rerank minimal 1
    top_n_rerank = max(1, int(top_n_rerank))

    if not nodes:
        return []

    # Graceful skip: model tidak ada di memori
    if _reranker_model is None:
        logger.warning(
            "[reranker] rerank() dipanggil tapi model tidak loaded → graceful skip, "
            "kembalikan %d node pertama tanpa scoring",
            min(top_n_rerank, len(nodes)),
        )
        return _build_output_nodes(list(nodes[:top_n_rerank]), scores=None)

    # Bangun pasangan (query, node.text) untuk CrossEncoder.predict()
    pairs = [(query, n.text) for n in nodes]

    try:
        # predict() mengembalikan array skor float, satu per pasangan
        raw_scores = _reranker_model.predict(pairs)
    except Exception as exc:
        logger.error(
            "[reranker] CrossEncoder.predict() gagal: %s → graceful skip", exc
        )
        return _build_output_nodes(list(nodes[:top_n_rerank]), scores=None)

    # Pasangkan skor dengan node, sort descending (skor tinggi = lebih relevan)
    float_scores = [float(s) for s in raw_scores]

    scored_pairs = sorted(
        zip(float_scores, nodes, strict=True),
        key=lambda x: x[0],
        reverse=True,
    )

    top_scored    = scored_pairs[:top_n_rerank]
    top_scores    = [s for s, _ in top_scored]
    top_nodes     = [n for _, n in top_scored]

    return _build_output_nodes(top_nodes, scores=top_scores)


def unload_reranker() -> None:
    """
    Unload Cross-Encoder dari memori dan bebaskan VRAM.

    WAJIB dipanggil setelah rerank() dan SEBELUM LLM dimuat.
    Aman dipanggil berkali-kali atau saat model tidak loaded (idempotent).
    """
    global _reranker_model, _reranker_model_name, _reranker_resolved_device
    global _reranker_requested_device

    if _reranker_model is None:
        logger.debug(
            "[reranker] unload_reranker() dipanggil tapi tidak ada model yang loaded → skip"
        )
        return

    logger.info(
        "[reranker] Unloading model '%s' dari device='%s'...",
        _reranker_model_name,
        _reranker_resolved_device,
    )
    prev_device = _reranker_resolved_device

    # Hapus referensi ke model — state tracking ikut direset
    _reranker_model = None
    _reranker_model_name = None
    _reranker_resolved_device = None
    _reranker_requested_device = None

    # Force garbage collection untuk membebaskan memori Python
    gc.collect()

    # Bebaskan CUDA memory cache jika model sebelumnya ada di GPU
    if prev_device == "cuda":
        try:
            import torch

            torch.cuda.empty_cache()
            logger.info("[reranker] CUDA cache berhasil di-clear")
        except ImportError:
            pass

    logger.info("[reranker] Model berhasil di-unload")


def is_reranker_loaded() -> bool:
    """
    Kembalikan True jika Cross-Encoder saat ini ada di memori dan siap dipakai.
    Gunakan ini untuk guard check di app_streamlit.py sebelum memanggil rerank().
    """
    return _reranker_model is not None


def get_reranker_info() -> Dict[str, Any]:
    """
    Kembalikan info status reranker untuk logging JSONL dan display UI.
    Shape selalu stabil (tidak ada KeyError di caller).

    Field:
        loaded           : bool — apakah model saat ini ada di memori
        model_name       : str | None — nama model yang loaded
        requested_device : str | None — device yang diminta config ("auto"|"cpu"|"cuda")
        resolved_device  : str | None — device yang benar-benar dipakai ("cpu"|"cuda")
        fallback_reason  : str | None — alasan fallback device jika resolved != cuda
        load_failed      : bool — apakah load() terakhir gagal
        error_msg        : str | None — pesan error jika load gagal
    """
    return {
        "loaded":            _reranker_model is not None,
        "model_name":        _reranker_model_name,
        "requested_device":  _reranker_requested_device,
        "resolved_device":   _reranker_resolved_device,
        "fallback_reason":   _load_fallback_reason,
        "load_failed":       _load_failed,
        "error_msg":         _load_error_msg,
    }


# ── Internal Helper ───────────────────────────────────────────────────────────

def _build_output_nodes(
    nodes: list[RetrievedNode],
    scores: Optional[List[float]],
) -> list[RetrievedNode]:
    """
    Bangun list[RetrievedNode] BARU dengan rerank_score dan rerank_rank terisi.

    NON-MUTATING: tidak memodifikasi input nodes sama sekali.
    Jika scores=None (graceful skip): rerank_score=None, rerank_rank=None.

    Args:
        nodes  : Node yang akan dikonversi ke output
        scores : Skor hasil CrossEncoder.predict() (sudah di-sort), atau None
    """
    output: list[RetrievedNode] = []
    for rank_idx, node in enumerate(nodes, start=1):
        score = scores[rank_idx - 1] if scores is not None else None
        new_node = dataclasses.replace(
            node,
            rerank_score=score,
            rerank_rank=rank_idx if scores is not None else None,
        )
        output.append(new_node)
    return output