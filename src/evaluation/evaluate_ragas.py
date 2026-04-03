from __future__ import annotations

import argparse
import copy
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from src.core.config import load_config


def _install_warning_filters() -> None:
    """
    4B.3:
    Bersihkan warning deprecation yang paling mengganggu di terminal evaluator.
    """
    patterns = [
        r".*LangchainLLMWrapper is deprecated.*",
        r".*LangchainEmbeddingsWrapper is deprecated.*",
        r".*Importing .* from 'ragas\.metrics' is deprecated.*",
    ]
    for pat in patterns:
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            message=pat,
        )

    # fallback lama tetap dipertahankan
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"ragas\..*",
    )


_install_warning_filters()


# ============================================================================
# IO helpers
# ============================================================================

def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_manifest_config(run_path: Path) -> Dict[str, Any]:
    manifest_path = run_path / "manifest.json"
    if not manifest_path.exists():
        return {}

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    cfg = data.get("config") or {}
    return cfg if isinstance(cfg, dict) else {}


def _read_eval_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"eval_dataset.json tidak ditemukan: {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Gagal membaca eval_dataset.json: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError("eval_dataset.json harus berbentuk list of dict rows.")

    out: List[Dict[str, Any]] = []
    for row in data:
        if isinstance(row, dict):
            out.append(row)

    return out


# ============================================================================
# Config helpers
# ============================================================================

def _cfg_get(cfg: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_int(x: Any, default: int) -> int:
    if x is None:
        return default
    try:
        return int(x)
    except Exception:
        return default


def _as_float(x: Any, default: float) -> float:
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _is_missing_override_value(
    x: Any,
    *,
    empty_string_is_missing: bool = True,
) -> bool:
    if x is None:
        return True
    if empty_string_is_missing and isinstance(x, str) and x.strip() == "":
        return True
    return False


def _first_present(
    *values: Any,
    empty_string_is_missing: bool = True,
) -> Any:
    """
    Ambil nilai pertama yang benar-benar hadir.
    Penting: 0, 0.0, False TIDAK dianggap missing.
    """
    for v in values:
        if not _is_missing_override_value(v, empty_string_is_missing=empty_string_is_missing):
            return v
    return None


def _effective_eval_llm_info(eval_llm_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "provider": str(eval_llm_cfg.get("provider", "(unknown)") or "(unknown)"),
        "model": str(eval_llm_cfg.get("model", "(unknown)") or "(unknown)"),
        "temperature": _to_float_or_none(eval_llm_cfg.get("temperature")),
        "max_tokens": _as_int(eval_llm_cfg.get("max_tokens"), 0),
        "timeout": _as_int(eval_llm_cfg.get("timeout"), 0),
        "max_retries": _as_int(eval_llm_cfg.get("max_retries"), 0),
        "override_enabled": bool(eval_llm_cfg.get("override_enabled", False)),
        "source": str(eval_llm_cfg.get("source", "unknown") or "unknown"),
        "config_origin": str(eval_llm_cfg.get("config_origin", "unknown") or "unknown"),
    }


def _build_eval_embeddings(cfg: Dict[str, Any]) -> HuggingFaceEmbeddings:
    model_name = str(
        _cfg_get(
            cfg,
            ["embeddings", "model_name"],
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    )
    device = str(_cfg_get(cfg, ["embeddings", "device"], "cpu"))
    normalize = bool(_cfg_get(cfg, ["embeddings", "normalize"], True))

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize},
    )


def _load_current_evaluation_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    4B.3:
    Evaluator boleh memakai config evaluator TERKINI dari file project sekarang,
    bukan hanya snapshot manifest run lama.

    Alasannya:
    - evaluation.llm_override adalah jalur evaluator-only
    - ia tidak mengubah runtime generator utama
    - memudahkan tuning evaluator tanpa harus membuat run baru hanya demi manifest baru
    """
    config_path = str(cfg.get("_config_path", "") or "").strip()
    base_path = str(cfg.get("_base_path", "configs/base.yaml") or "configs/base.yaml").strip()

    if not config_path:
        return {}

    try:
        current_cfg = load_config(config_path, base_path=base_path)
    except Exception:
        return {}

    eval_root = current_cfg.get("evaluation") or {}
    return eval_root if isinstance(eval_root, dict) else {}


def _effective_eval_root(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prioritas root config evaluator:
    1) config project TERKINI
    2) fallback ke snapshot manifest
    """
    current_eval = _load_current_evaluation_config(cfg)
    if current_eval:
        return current_eval

    manifest_eval = cfg.get("evaluation") or {}
    return manifest_eval if isinstance(manifest_eval, dict) else {}


def _resolve_eval_llm_cfg(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Resolve konfigurasi LLM evaluator dengan prioritas:

    1) CLI args evaluator
    2) env vars evaluator (EVAL_*)
    3) cfg["evaluation"]["llm_override"] TERKINI jika enabled=true
    4) env vars runtime lama
    5) cfg["llm"] runtime lama

    4B.3:
    - nilai seperti 0.0 tidak boleh hilang hanya karena falsy
    - evaluator default boleh memakai config evaluator TERKINI project,
        bukan hanya snapshot manifest lama
    """
    load_dotenv()

    runtime_llm_cfg = cfg.get("llm") or {}
    eval_root = _effective_eval_root(cfg)
    eval_llm_cfg = eval_root.get("llm_override") or {}

    override_enabled = bool(eval_llm_cfg.get("enabled", False))
    config_origin = "current_project_config" if _load_current_evaluation_config(cfg) else "manifest_snapshot"

    def _cfg_eval(key: str) -> Any:
        if override_enabled:
            val = eval_llm_cfg.get(key, None)
            if not _is_missing_override_value(val):
                return val
        return None

    provider = str(
        _first_present(
            getattr(args, "eval_provider", None),
            os.getenv("EVAL_LLM_PROVIDER"),
            _cfg_eval("provider"),
            os.getenv("LLM_PROVIDER"),
            runtime_llm_cfg.get("provider"),
            "groq",
        )
        or "groq"
    ).strip().lower()

    runtime_model_fallback = str(runtime_llm_cfg.get("model", "") or "").strip()
    cli_eval_model = getattr(args, "eval_model", None)
    env_eval_model = os.getenv("EVAL_LLM_MODEL")
    cfg_eval_model = _cfg_eval("model")

    if provider == "groq":
        model = str(
            _first_present(
                cli_eval_model,
                os.getenv("EVAL_GROQ_MODEL"),
                env_eval_model,
                cfg_eval_model,
                os.getenv("GROQ_MODEL"),
                runtime_model_fallback,
                "llama-3.1-8b-instant",
            )
            or "llama-3.1-8b-instant"
        ).strip()
    elif provider == "openai":
        model = str(
            _first_present(
                cli_eval_model,
                os.getenv("EVAL_OPENAI_MODEL"),
                env_eval_model,
                cfg_eval_model,
                os.getenv("OPENAI_MODEL"),
                runtime_model_fallback,
                "gpt-4o-mini",
            )
            or "gpt-4o-mini"
        ).strip()
    elif provider == "ollama":
        model = str(
            _first_present(
                cli_eval_model,
                os.getenv("EVAL_OLLAMA_MODEL"),
                env_eval_model,
                cfg_eval_model,
                os.getenv("OLLAMA_MODEL"),
                runtime_model_fallback,
                "llama3",
            )
            or "llama3"
        ).strip()
    else:
        model = str(
            _first_present(
                cli_eval_model,
                env_eval_model,
                cfg_eval_model,
                runtime_model_fallback,
                "(unknown)",
            )
            or "(unknown)"
        ).strip()

    temperature = _as_float(
        _first_present(
            getattr(args, "eval_temperature", None),
            os.getenv("EVAL_LLM_TEMPERATURE"),
            _cfg_eval("temperature"),
            runtime_llm_cfg.get("temperature", 0.2),
        ),
        0.2,
    )

    max_tokens = _as_int(
        _first_present(
            getattr(args, "eval_max_tokens", None),
            os.getenv("EVAL_LLM_MAX_TOKENS"),
            _cfg_eval("max_tokens"),
            runtime_llm_cfg.get("max_tokens", 512),
        ),
        512,
    )

    timeout = _as_int(
        _first_present(
            getattr(args, "eval_timeout", None),
            os.getenv("EVAL_LLM_TIMEOUT"),
            _cfg_eval("timeout"),
            runtime_llm_cfg.get("timeout", 60),
        ),
        60,
    )

    max_retries = _as_int(
        _first_present(
            getattr(args, "eval_max_retries", None),
            os.getenv("EVAL_LLM_MAX_RETRIES"),
            _cfg_eval("max_retries"),
            runtime_llm_cfg.get("max_retries", 2),
        ),
        2,
    )

    ollama_base_url = str(
        _first_present(
            getattr(args, "eval_ollama_base_url", None),
            os.getenv("EVAL_OLLAMA_BASE_URL"),
            _cfg_eval("ollama_base_url"),
            os.getenv("OLLAMA_BASE_URL"),
            "http://localhost:11434",
        )
        or "http://localhost:11434"
    ).strip()

    cli_override_used = any(
        getattr(args, name, None) is not None
        for name in (
            "eval_provider",
            "eval_model",
            "eval_temperature",
            "eval_max_tokens",
            "eval_timeout",
            "eval_max_retries",
            "eval_ollama_base_url",
        )
    )

    env_override_used = any(
        os.getenv(name) not in (None, "")
        for name in (
            "EVAL_LLM_PROVIDER",
            "EVAL_LLM_MODEL",
            "EVAL_GROQ_MODEL",
            "EVAL_OPENAI_MODEL",
            "EVAL_OLLAMA_MODEL",
            "EVAL_LLM_TEMPERATURE",
            "EVAL_LLM_MAX_TOKENS",
            "EVAL_LLM_TIMEOUT",
            "EVAL_LLM_MAX_RETRIES",
            "EVAL_OLLAMA_BASE_URL",
        )
    )

    source = "runtime_llm_default"
    if override_enabled:
        source = "evaluation.llm_override"
    if env_override_used:
        source = "env_eval"
    if cli_override_used:
        source = "cli_eval"

    return {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
        "max_retries": max_retries,
        "ollama_base_url": ollama_base_url,
        "override_enabled": override_enabled,
        "source": source,
        "config_origin": config_origin,
    }


def _resolve_eval_batch_size(cfg: Dict[str, Any], args: argparse.Namespace) -> int:
    """
    4B.3:
    Batch size default diambil dari config evaluator TERKINI.
    Terminal (CLI) tetap boleh override.
    """
    eval_root = _effective_eval_root(cfg)
    ragas_cfg = eval_root.get("ragas") or {}

    batch_size = _as_int(
        _first_present(
            getattr(args, "batch_size", None),
            os.getenv("EVAL_RAGAS_BATCH_SIZE"),
            ragas_cfg.get("batch_size"),
            1,
        ),
        1,
    )
    return max(batch_size, 1)


def _build_eval_llm(eval_llm_cfg: Dict[str, Any]):
    """
    Build ChatModel khusus evaluator RAGAS.
    Tidak membaca cfg["llm"] lagi secara langsung.
    """
    provider = str(eval_llm_cfg.get("provider", "groq") or "groq").strip().lower()
    model = str(eval_llm_cfg.get("model", "") or "").strip()
    temperature = _as_float(eval_llm_cfg.get("temperature"), 0.2)
    max_tokens = _as_int(eval_llm_cfg.get("max_tokens"), 512)
    timeout = _as_int(eval_llm_cfg.get("timeout"), 60)
    max_retries = _as_int(eval_llm_cfg.get("max_retries"), 2)

    if provider == "groq":
        from langchain_groq import ChatGroq  # type: ignore

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY belum di-set untuk evaluator RAGAS.")

        return ChatGroq(
            model=model or "llama-3.1-8b-instant",
            temperature=temperature,
            timeout=timeout,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI  # type: ignore

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY belum di-set untuk evaluator RAGAS.")

        kwargs_new: Dict[str, Any] = {
            "model": model or "gpt-4o-mini",
            "temperature": temperature,
            "timeout": timeout,
            "max_retries": max_retries,
            "model_kwargs": {"max_tokens": max_tokens},
        }

        try:
            return ChatOpenAI(**kwargs_new)
        except TypeError:
            kwargs_old: Dict[str, Any] = {
                "model_name": model or "gpt-4o-mini",
                "temperature": temperature,
                "request_timeout": timeout,
                "max_retries": max_retries,
                "max_tokens": max_tokens,
            }
            try:
                return ChatOpenAI(**kwargs_old)
            except TypeError:
                return ChatOpenAI()

    if provider == "ollama":
        from langchain_ollama import ChatOllama  # type: ignore

        return ChatOllama(
            base_url=str(eval_llm_cfg.get("ollama_base_url") or "http://localhost:11434"),
            model=model or "llama3",
            temperature=temperature,
            # optional: model_kwargs={"num_predict": max_tokens},
        )

    raise ValueError(
        f"Provider evaluator tidak dikenal: {provider}. Gunakan: groq | openai | ollama"
    )


# ============================================================================
# Dataset normalization
# ============================================================================

def _normalize_text_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []

    out: List[str] = []
    for x in values:
        s = str(x or "").strip()
        if s:
            out.append(s)
    return out


def _normalize_eval_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Siapkan row agar kompatibel lintas versi RAGAS.

    sengaja menyimpan DUA shape sekaligus:
    - shape lama  : question / answer / contexts
    - shape baru  : user_input / response / retrieved_contexts

    Dengan begitu evaluator lebih tahan terhadap perubahan schema RAGAS.
    """
    out: List[Dict[str, Any]] = []

    for row in rows:
        q = str(
            row.get("question")
            or row.get("question_eval")
            or row.get("question_user")
            or ""
        ).strip()
        a = str(row.get("answer", "") or "").strip()
        contexts = _normalize_text_list(row.get("contexts", []))

        item = dict(row)
        item["question"] = q
        item["answer"] = a
        item["contexts"] = contexts

        # shape yang lebih baru / lebih umum dipakai di RAGAS modern
        item["user_input"] = q
        item["response"] = a
        item["retrieved_contexts"] = contexts

        # optional passthrough jika suatu hari dataset builder menyimpan reference
        ref = str(row.get("reference", "") or "").strip()
        if ref:
            item["reference"] = ref

        ref_contexts = _normalize_text_list(row.get("reference_contexts", []))
        if ref_contexts:
            item["reference_contexts"] = ref_contexts

        out.append(item)

    return out


def _dataset_has_reference_answers(rows: List[Dict[str, Any]]) -> bool:
    return all(bool(str(r.get("reference", "") or "").strip()) for r in rows) if rows else False


def _dataset_has_reference_contexts(rows: List[Dict[str, Any]]) -> bool:
    if not rows:
        return False

    for r in rows:
        ctxs = r.get("reference_contexts", [])
        if not isinstance(ctxs, list) or not any(str(x or "").strip() for x in ctxs):
            return False
    return True


def _make_ragas_dataset(rows: List[Dict[str, Any]]) -> Tuple[Any, str]:
    """
    Buat dataset evaluator dengan strategi kompatibilitas:

    1) pakai EvaluationDataset jikalau tersedia
    2) fallback ke HuggingFace Dataset
    """
    try:
        from ragas.dataset_schema import EvaluationDataset  # type: ignore
        return EvaluationDataset.from_list(rows), "ragas.EvaluationDataset"
    except Exception:
        pass

    try:
        from ragas import EvaluationDataset  # type: ignore
        return EvaluationDataset.from_list(rows), "ragas.EvaluationDataset"
    except Exception:
        pass

    from datasets import Dataset  # type: ignore
    return Dataset.from_list(rows), "datasets.Dataset"


# ============================================================================
# RAGAS wrappers / compatibility helpers
# ============================================================================

def _maybe_wrap_llm_for_ragas(llm: Any) -> Any:
    """
    Wrapper opsional untuk kompatibilitas lintas versi RAGAS.
    Warning deprecation ditahan lokal agar terminal tetap bersih.
    """
    if llm is None:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.llms import LangchainLLMWrapper  # type: ignore
            return LangchainLLMWrapper(llm)
    except Exception:
        pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.llms.base import LangchainLLMWrapper  # type: ignore
            return LangchainLLMWrapper(llm)
    except Exception:
        pass

    return llm


def _maybe_wrap_embeddings_for_ragas(embeddings: Any) -> Any:
    """
    Wrapper opsional untuk kompatibilitas lintas versi RAGAS.
    Warning deprecation ditahan lokal agar terminal tetap bersih.
    """
    if embeddings is None:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore
            return LangchainEmbeddingsWrapper(embeddings)
    except Exception:
        pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.embeddings.base import LangchainEmbeddingsWrapper  # type: ignore
            return LangchainEmbeddingsWrapper(embeddings)
    except Exception:
        pass

    return embeddings


def _safe_set_metric_attr(metric: Any, attr_name: str, value: Any) -> None:
    try:
        if hasattr(metric, attr_name) and getattr(metric, attr_name, None) is None:
            setattr(metric, attr_name, value)
    except Exception:
        pass


def _force_metric_single_generation(metric: Any) -> Any:
    """
    Paksa metric evaluator menjadi single-generation friendly.

    Penting untuk provider seperti Groq yang menolak n > 1.
    Target utama:
    - Answer Relevancy / Response Relevancy
    - beberapa metric lain yang punya strictness / n
    """
    if metric is None:
        return metric

    # Banyak metric RAGAS memakai strictness untuk jumlah variasi pertanyaan/judgement.
    for attr_name in ("strictness", "n", "num_generations"):
        try:
            if hasattr(metric, attr_name):
                setattr(metric, attr_name, 1)
        except Exception:
            pass

    # Beberapa metric menyimpan generator internal
    for child_name in (
        "question_generation",
        "statement_generator",
        "claim_decomposition_prompt",
    ):
        try:
            child = getattr(metric, child_name, None)
        except Exception:
            child = None

        if child is None:
            continue

        for attr_name in ("strictness", "n", "num_generations"):
            try:
                if hasattr(child, attr_name):
                    setattr(child, attr_name, 1)
            except Exception:
                pass

    return metric


def _import_metric_class(
    class_name: str,
    *,
    prefer_collections: bool = True,
) -> Optional[Any]:
    """
    Import helper untuk metric class RAGAS secara kompatibel.

    Urutan:
    1) ragas.metrics.collections
    2) ragas.metrics
    """
    if prefer_collections:
        try:
            from ragas.metrics import collections as ragas_metric_collections  # type: ignore
            if hasattr(ragas_metric_collections, class_name):
                return getattr(ragas_metric_collections, class_name)
        except Exception:
            pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas import metrics as ragas_metrics  # type: ignore
            if hasattr(ragas_metrics, class_name):
                return getattr(ragas_metrics, class_name)
    except Exception:
        pass

    return None


# ============================================================================
# Metric builders
# ============================================================================

def _build_faithfulness_metric(evaluator_llm: Any) -> Tuple[Optional[Any], Optional[str]]:
    try:
        from ragas.metrics.collections import faithfulness as faithfulness_metric  # type: ignore
        m = copy.deepcopy(faithfulness_metric)
        _safe_set_metric_attr(m, "llm", evaluator_llm)
        m = _force_metric_single_generation(m)
        return m, None
    except Exception:
        pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import faithfulness as faithfulness_metric  # type: ignore
        m = copy.deepcopy(faithfulness_metric)
        _safe_set_metric_attr(m, "llm", evaluator_llm)
        m = _force_metric_single_generation(m)
        return m, None
    except Exception:
        pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import Faithfulness  # type: ignore
        try:
            m = Faithfulness(llm=evaluator_llm)
            m = _force_metric_single_generation(m)
            return m, None
        except TypeError:
            m = Faithfulness()
            _safe_set_metric_attr(m, "llm", evaluator_llm)
            m = _force_metric_single_generation(m)
            return m, None
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}"


def _build_answer_relevancy_metric(
    evaluator_llm: Any,
    evaluator_embeddings: Any,
) -> Tuple[Optional[Any], Optional[str]]:
    try:
        from ragas.metrics.collections import (
            answer_relevancy as answer_relevancy_metric,  # type: ignore
        )
        m = copy.deepcopy(answer_relevancy_metric)
        _safe_set_metric_attr(m, "llm", evaluator_llm)
        _safe_set_metric_attr(m, "embeddings", evaluator_embeddings)
        m = _force_metric_single_generation(m)   # ← PATCH KUNCI UNTUK GROQ
        return m, None
    except Exception:
        pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.metrics import answer_relevancy as answer_relevancy_metric  # type: ignore
        m = copy.deepcopy(answer_relevancy_metric)
        _safe_set_metric_attr(m, "llm", evaluator_llm)
        _safe_set_metric_attr(m, "embeddings", evaluator_embeddings)
        m = _force_metric_single_generation(m)  # ← PATCH KUNCI UNTUK GROQ
        return m, None
    except Exception:
        pass

    for class_name in ("ResponseRelevancy", "AnswerRelevancy"):
        try:
            from ragas import metrics as ragas_metrics  # type: ignore
            cls = getattr(ragas_metrics, class_name)

            try:
                m = cls(llm=evaluator_llm, embeddings=evaluator_embeddings)
                m = _force_metric_single_generation(m)
                return m, None
            except TypeError:
                m = cls()
                _safe_set_metric_attr(m, "llm", evaluator_llm)
                _safe_set_metric_attr(m, "embeddings", evaluator_embeddings)
                m = _force_metric_single_generation(m)
                return m, None
        except Exception:
            continue

    return None, "Metric answer relevancy/response relevancy tidak ditemukan pada versi RAGAS aktif."


def _build_context_precision_metric(
    rows: List[Dict[str, Any]],
    evaluator_llm: Any,
    evaluator_embeddings: Any,
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Strategi:
    - jika ada reference_contexts   -> coba metric reference-based modern
    - jika ada reference answer     -> coba with-reference
    - selain itu                    -> prefer without-reference
    - jika varian yang cocok tidak tersedia di versi RAGAS aktif, skip secara jujur
    """
    has_ref_contexts = _dataset_has_reference_contexts(rows)
    has_ref_answers = _dataset_has_reference_answers(rows)

    # A) Modern / stable-ish: reference_contexts-based
    if has_ref_contexts:
        for class_name in ("ContextPrecision", "NonLLMContextPrecisionWithReference"):
            cls = _import_metric_class(class_name)
            if cls is None:
                continue

            try:
                m = cls(embeddings=evaluator_embeddings)
                m = _force_metric_single_generation(m)
                return m, None
            except TypeError:
                try:
                    m = cls()
                    _safe_set_metric_attr(m, "embeddings", evaluator_embeddings)
                    m = _force_metric_single_generation(m)
                    return m, None
                except Exception:
                    continue
            except Exception:
                continue

        return None, (
            "Dataset memiliki reference_contexts, tetapi metric context precision "
            "reference-based tidak ditemukan pada versi RAGAS aktif."
        )

    # B) Reference-answer based
    if has_ref_answers:
        for class_name in ("LLMContextPrecisionWithReference",):
            cls = _import_metric_class(class_name)
            if cls is None:
                continue

            try:
                m = cls(llm=evaluator_llm)
                m = _force_metric_single_generation(m)
                return m, None
            except TypeError:
                try:
                    m = cls()
                    _safe_set_metric_attr(m, "llm", evaluator_llm)
                    m = _force_metric_single_generation(m)
                    return m, None
                except Exception:
                    continue
            except Exception:
                continue

        return None, (
            "Dataset memiliki reference answer, tetapi metric "
            "LLMContextPrecisionWithReference tidak tersedia pada versi RAGAS aktif."
        )

    # C) No-reference (paling cocok untuk kondisi dataset saat ini)
    for class_name in ("LLMContextPrecisionWithoutReference",):
        cls = _import_metric_class(class_name)
        if cls is None:
            continue

        try:
            m = cls(llm=evaluator_llm)
            m = _force_metric_single_generation(m)
            return m, None
        except TypeError:
            try:
                m = cls()
                _safe_set_metric_attr(m, "llm", evaluator_llm)
                m = _force_metric_single_generation(m)
                return m, None
            except Exception:
                continue
        except Exception:
            continue

    return None, (
        "Metric context precision tanpa reference tidak tersedia pada versi RAGAS aktif. "
        "Dataset Phase 4 saat ini juga belum menyimpan reference/reference_contexts, "
        "jadi metric ini di-skip secara jujur."
    )


def _build_metrics(
    rows: List[Dict[str, Any]],
    evaluator_llm: Any,
    evaluator_embeddings: Any,
) -> Tuple[List[Any], List[str], Dict[str, str]]:
    metrics: List[Any] = []
    metric_names: List[str] = []
    skipped: Dict[str, str] = {}

    faithfulness_metric, faithfulness_err = _build_faithfulness_metric(evaluator_llm)
    if faithfulness_metric is not None:
        metrics.append(faithfulness_metric)
        metric_names.append("faithfulness")
    else:
        skipped["faithfulness"] = faithfulness_err or "unknown_error"

    answer_rel_metric, answer_rel_err = _build_answer_relevancy_metric(
        evaluator_llm,
        evaluator_embeddings,
    )
    if answer_rel_metric is not None:
        metrics.append(answer_rel_metric)
        metric_names.append("answer_relevancy")
    else:
        skipped["answer_relevancy"] = answer_rel_err or "unknown_error"

    context_precision_metric, context_precision_err = _build_context_precision_metric(
        rows,
        evaluator_llm,
        evaluator_embeddings,
    )
    if context_precision_metric is not None:
        metrics.append(context_precision_metric)
        metric_names.append("context_precision")
    else:
        skipped["context_precision"] = context_precision_err or "unknown_error"

    return metrics, metric_names, skipped


# ============================================================================
# Evaluation execution
# ============================================================================

def _run_ragas_evaluation(
    *,
    dataset: Any,
    metrics: List[Any],
    evaluator_llm: Any,
    evaluator_embeddings: Any,
    raise_exceptions: bool,
    batch_size: int,
) -> Any:
    from ragas import evaluate  # type: ignore

    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "metrics": metrics,
        "llm": evaluator_llm,
        "embeddings": evaluator_embeddings,
        "raise_exceptions": raise_exceptions,
        "batch_size": max(int(batch_size), 1),
    }

    try:
        return evaluate(**kwargs)
    except TypeError:
        # fallback untuk versi yang signature-nya lebih sempit
        kwargs.pop("embeddings", None)
        try:
            return evaluate(**kwargs)
        except TypeError:
            # fallback paling konservatif
            kwargs.pop("batch_size", None)
            return evaluate(**kwargs)


def _result_to_pandas(result: Any) -> pd.DataFrame:
    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        if isinstance(df, pd.DataFrame):
            return df

    if isinstance(result, pd.DataFrame):
        return result

    raise TypeError(
        "Hasil evaluasi RAGAS tidak bisa dikonversi ke pandas DataFrame. "
        "Periksa versi RAGAS aktif."
    )


# ============================================================================
# Result normalization
# ============================================================================

_METRIC_COLUMN_ALIASES: Dict[str, List[str]] = {
    "faithfulness": [
        "faithfulness",
    ],
    "answer_relevancy": [
        "answer_relevancy",
        "answer_relevance",
        "response_relevancy",
    ],
    "context_precision": [
        "context_precision",
        "llm_context_precision_without_reference",
        "context_precision_without_reference",
        "llm_context_precision_with_reference",
        "context_precision_with_reference",
        "non_llm_context_precision_with_reference",
    ],
}


_LOGICAL_METRICS: Tuple[str, ...] = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
)


def _resolve_metric_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)
    lower_map = {str(c).lower(): str(c) for c in cols}

    out: Dict[str, str] = {}
    for logical_name, aliases in _METRIC_COLUMN_ALIASES.items():
        for alias in aliases:
            key = alias.lower()
            if key in lower_map:
                out[logical_name] = lower_map[key]
                break
    return out


def _to_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        fx = float(x)
        if math.isnan(fx):
            return None
        return fx
    except Exception:
        return None


def _mean_numeric(series: pd.Series) -> Optional[float]:
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals.dropna()
    if len(vals) == 0:
        return None
    return float(vals.mean())


def _metric_value_from_row(
    row_df: pd.Series,
    metric_columns: Dict[str, str],
    logical_name: str,
) -> Optional[float]:
    """
    Ambil nilai metric per-row secara aman.
    """
    col = metric_columns.get(logical_name)
    if not col or col not in row_df.index:
        return None
    return _to_float_or_none(row_df[col])


def _metric_status_from_value(v: Optional[float]) -> str:
    """
    Status sederhana per metric:
    - ok      -> ada skor numerik
    - missing -> null / NaN / tidak tersedia
    """
    return "ok" if v is not None else "missing"


def _build_metric_status_map(
    row_df: pd.Series,
    metric_columns: Dict[str, str],
) -> Tuple[Dict[str, Optional[float]], Dict[str, str], List[str]]:
    """
    Return:
    - metric_values  : logical metric -> float | None
    - metric_status  : logical metric -> "ok" | "missing"
    - missing_metrics: list nama metric yang missing
    """
    metric_values: Dict[str, Optional[float]] = {}
    metric_status: Dict[str, str] = {}
    missing_metrics: List[str] = []

    for logical_name in _LOGICAL_METRICS:
        val = _metric_value_from_row(row_df, metric_columns, logical_name)
        metric_values[logical_name] = val
        metric_status[logical_name] = _metric_status_from_value(val)
        if val is None:
            missing_metrics.append(logical_name)

    return metric_values, metric_status, missing_metrics


def _count_missing_metrics(item_rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Hitung berapa row yang missing untuk tiap metric logical.
    """
    out = {m: 0 for m in _LOGICAL_METRICS}
    for row in item_rows:
        for m in _LOGICAL_METRICS:
            if row.get(m) is None:
                out[m] += 1
    return out


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument(
        "--raise_exceptions",
        action="store_true",
        help="Jika aktif, RAGAS akan raise exception per-row bila metric gagal.",
    )

    # ── Phase 4B.2: evaluator-specific LLM override ─────────────────────────
    ap.add_argument("--eval_provider", choices=["groq", "openai", "ollama"])
    ap.add_argument("--eval_model", default=None)
    ap.add_argument("--eval_temperature", type=float, default=None)
    ap.add_argument("--eval_max_tokens", type=int, default=None)
    ap.add_argument("--eval_timeout", type=int, default=None)
    ap.add_argument("--eval_max_retries", type=int, default=None)
    ap.add_argument("--eval_ollama_base_url", default=None)

    ap.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size evaluator RAGAS. Jika tidak diisi, ambil dari evaluation.ragas.batch_size.",
    )
    args = ap.parse_args()

    run_path = Path(args.runs_dir) / args.run_id
    dataset_path = run_path / "eval_dataset.json"

    run_cfg = _load_manifest_config(run_path)
    raw_rows = _read_eval_rows(dataset_path)
    eval_rows = _normalize_eval_rows(raw_rows)
    effective_batch_size = _resolve_eval_batch_size(run_cfg, args)

    if not eval_rows:
        raise ValueError("eval_dataset.json kosong - tidak ada row untuk dievaluasi.")

    # Phase 4B.2:
    # evaluator sekarang memilki jalur konfigurasi LLM sendiri,
    # terpisah dari runtime generator utama.
    eval_llm_cfg = _resolve_eval_llm_cfg(run_cfg, args)
    evaluator_llm = _build_eval_llm(eval_llm_cfg)
    evaluator_embeddings = _build_eval_embeddings(run_cfg)

    # Bungkus ke wrapper RAGAS jika versi aktif menyediakannya.
    evaluator_llm_for_ragas = _maybe_wrap_llm_for_ragas(evaluator_llm)
    evaluator_embeddings_for_ragas = _maybe_wrap_embeddings_for_ragas(evaluator_embeddings)

    dataset, dataset_backend = _make_ragas_dataset(eval_rows)

    metrics, metric_names, skipped_metrics = _build_metrics(
        eval_rows,
        evaluator_llm_for_ragas,
        evaluator_embeddings_for_ragas,
    )

    if not metrics:
        raise RuntimeError(
            "Tidak ada metric RAGAS yang berhasil dibangun. "
            f"Detail skip: {skipped_metrics}"
        )

    result = _run_ragas_evaluation(
        dataset=dataset,
        metrics=metrics,
        evaluator_llm=evaluator_llm_for_ragas,
        evaluator_embeddings=evaluator_embeddings_for_ragas,
        raise_exceptions=bool(args.raise_exceptions),
        batch_size=effective_batch_size,
    )

    df = _result_to_pandas(result)
    metric_columns = _resolve_metric_columns(df)

    # Bangun summary agregat
    metrics_summary: Dict[str, Optional[float]] = {}
    for logical_name in _LOGICAL_METRICS:
        col = metric_columns.get(logical_name)
        if col and col in df.columns:
            metrics_summary[logical_name] = _mean_numeric(df[col])
        else:
            metrics_summary[logical_name] = None

    # Bangun hasil per-item yang mudah diaudit
    item_rows: List[Dict[str, Any]] = []
    n_join_rows = min(len(eval_rows), len(df))

    for idx in range(n_join_rows):
        base = dict(eval_rows[idx])
        row_df = df.iloc[idx]

        metric_values, metric_status, missing_metrics = _build_metric_status_map(
            row_df,
            metric_columns,
        )

        row_has_missing_metrics = bool(missing_metrics)
        row_status = "partial" if row_has_missing_metrics else "ok"

        item: Dict[str, Any] = {
            "run_id": base.get("run_id"),
            "turn_id": base.get("turn_id"),
            "question": base.get("question"),
            "question_user": base.get("question_user"),
            "question_eval": base.get("question_eval"),
            "question_source": base.get("question_source"),
            "answer": base.get("answer"),
            "context_count": base.get("context_count"),
            "strategy": base.get("strategy"),
            "retrieval_mode": base.get("retrieval_mode"),
            "generator_status": base.get("generator_status"),
            "verification_label": base.get("verification_label"),
            "verification_score": base.get("verification_score"),
            "used_context_chunk_ids": base.get("used_context_chunk_ids"),
            "ragas_row_index": idx,

            # ← 4B.1 transparency fields
            "metric_status": metric_status,
            "missing_metrics": missing_metrics,
            "row_has_missing_metrics": row_has_missing_metrics,
            "row_status": row_status,
            "partial_failure_note": (
                "Sebagian metric bernilai null; kemungkinan timeout atau internal metric failure."
                if row_has_missing_metrics
                else None
            ),
        }

        # Pertahankan field flat lama agar backward-friendly
        for logical_name in _LOGICAL_METRICS:
            item[logical_name] = metric_values.get(logical_name)

        item_rows.append(item)

    metrics_missing_counts = _count_missing_metrics(item_rows)
    rows_with_missing_metrics = sum(
        1 for row in item_rows if bool(row.get("row_has_missing_metrics"))
    )
    rows_all_metrics_present = len(item_rows) - rows_with_missing_metrics
    partial_failures_detected = rows_with_missing_metrics > 0

    llm_info = _effective_eval_llm_info(eval_llm_cfg)
    emb_model_name = str(
        _cfg_get(
            run_cfg,
            ["embeddings", "model_name"],
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    )

    results_payload: Dict[str, Any] = {
        "run_id": args.run_id,
        "dataset_path": str(dataset_path),
        "dataset_backend": dataset_backend,
        "n_input_rows": len(raw_rows),
        "n_eval_rows": len(eval_rows),
        "n_result_rows": int(len(df)),
        "n_item_rows_written": len(item_rows),
        "row_count_mismatch": bool(len(eval_rows) != len(df)),
        "llm_provider": llm_info["provider"],
        "llm_model": llm_info["model"],
        "embedding_model": emb_model_name,

        # ── Phase 4B.2: evaluator-specific transparency ───────────────────────
        "evaluator_llm_provider": llm_info["provider"],
        "evaluator_llm_model": llm_info["model"],
        "evaluator_llm_temperature": llm_info["temperature"],
        "evaluator_llm_max_tokens": llm_info["max_tokens"],
        "evaluator_llm_timeout": llm_info["timeout"],
        "evaluator_llm_max_retries": llm_info["max_retries"],
        "evaluator_llm_override_enabled": llm_info["override_enabled"],
        "evaluator_llm_source": llm_info["source"],
        "evaluator_llm_config_origin": llm_info["config_origin"],
        "batch_size": effective_batch_size,

        "metrics_requested": list(_LOGICAL_METRICS),
        "metrics_executed": metric_names,
        "metrics_skipped": skipped_metrics,
        "metric_columns": metric_columns,
        "metrics_summary": metrics_summary,

        # ← 4B.1 transparency summary
        "metrics_missing_counts": metrics_missing_counts,
        "rows_with_missing_metrics": rows_with_missing_metrics,
        "rows_all_metrics_present": rows_all_metrics_present,
        "partial_failures_detected": partial_failures_detected,
        "evaluation_note": (
            "Sebagian row memiliki metric null; hasil agregat dihitung hanya dari nilai yang tersedia."
            if partial_failures_detected
            else "Semua row memiliki skor untuk seluruh metric yang dieksekusi."
        ),
    }

    results_path = run_path / "ragas_results.json"
    item_scores_path = run_path / "ragas_item_scores.jsonl"
    item_scores_pretty_path = run_path / "ragas_item_scores.pretty.json"

    results_path.write_text(
        json.dumps(results_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(item_scores_path, item_rows)
    item_scores_pretty_path.write_text(
        json.dumps(item_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved RAGAS summary : {results_path}")
    print(f"Saved RAGAS items   : {item_scores_path}")
    print(f"Saved RAGAS pretty  : {item_scores_pretty_path}")
    print(f"Dataset backend     : {dataset_backend}")
    print(f"Rows evaluated      : {len(eval_rows)}")
    print(f"Batch size          : {effective_batch_size}")
    print(f"Metrics executed    : {', '.join(metric_names) if metric_names else '(none)'}")
    print(f"Rows partial/missing: {rows_with_missing_metrics}")

    print(f"Evaluator provider  : {llm_info['provider']}")
    print(f"Evaluator model     : {llm_info['model']}")
    print(f"Evaluator source    : {llm_info['source']}")
    print(f"Evaluator cfg origin: {llm_info['config_origin']}")
    print(f"Evaluator temp      : {llm_info['temperature']}")
    print(f"Evaluator max_tokens: {llm_info['max_tokens']}")
    print(f"Evaluator timeout   : {llm_info['timeout']}")
    print(f"Evaluator retries   : {llm_info['max_retries']}")

    for metric_name in _LOGICAL_METRICS:
        val = results_payload["metrics_summary"].get(metric_name)
        miss = results_payload["metrics_missing_counts"].get(metric_name, 0)
        if val is None:
            print(f"- {metric_name}: (not available) | missing_rows={miss}")
        else:
            print(f"- {metric_name}: {val:.6f} | missing_rows={miss}")

    if partial_failures_detected:
        print("NOTE: sebagian row memiliki metric null; cek ragas_item_scores.pretty.json untuk audit detail.")


if __name__ == "__main__":
    main()