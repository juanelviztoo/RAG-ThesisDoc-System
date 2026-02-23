from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path
from typing import Iterable, Optional, Sequence

import streamlit as st
import streamlit.components.v1 as components


@st.cache_data(show_spinner=False)
def read_bytes_cached(path_str: str) -> bytes:
    return Path(path_str).read_bytes()


@st.cache_data(show_spinner=False)
def read_text_cached(path_str: str) -> str:
    return Path(path_str).read_text(encoding="utf-8")


def download_file_button(
    label: str,
    path: Path,
    mime: str,
    file_name: Optional[str] = None,
    *,
    disabled_caption: Optional[str] = None,
    key: Optional[str] = None,
) -> None:
    """Download button yang aman (kalau file belum ada, tombol disabled)."""
    if not path.exists():
        st.button(label, disabled=True, key=(key or f"disabled_{label}_{path.name}"))
        if disabled_caption:
            st.caption(disabled_caption)
        return

    data = read_bytes_cached(str(path))
    st.download_button(
        label=label,
        data=data,
        file_name=file_name or path.name,
        mime=mime,
        width="stretch",
        key=key,  # <-- PENTING
    )


def download_logs_zip_button(
    label: str,
    files: Iterable[Path],
    zip_name: str = "logs.zip",
) -> None:
    existing = [p for p in files if p.exists()]
    if not existing:
        st.button(label, disabled=True)
        st.caption("Belum ada file log untuk diunduh pada run ini.")
        return

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in existing:
            zf.writestr(p.name, read_bytes_cached(str(p)))
    buf.seek(0)

    st.download_button(
        label=label,
        data=buf.getvalue(),
        file_name=zip_name,
        mime="application/zip",
        width="stretch",
    )


def copy_to_clipboard_button(text: str, *, label: str = "📑", key: str) -> None:
    """
    Tombol copy clipboard via HTML/JS.
    """
    b64 = base64.b64encode((text or "").encode("utf-8")).decode("utf-8")
    html = f"""
    <div style="display:flex; justify-content:flex-end;">
      <button id="{key}" title="Copy" style="
        padding:6px 10px; border-radius:10px; border:1px solid #444; background:#111;
        color:#fff; cursor:pointer; font-size:14px;">
        {label}
      </button>
    </div>
    <script>
      const btn = document.getElementById("{key}");
      btn.addEventListener("click", async () => {{
        try {{
          const txt = atob("{b64}");
          await navigator.clipboard.writeText(txt);
          btn.textContent = "✅";
          setTimeout(() => btn.textContent = "{label}", 900);
        }} catch (e) {{
          btn.textContent = "⚠️";
          setTimeout(() => btn.textContent = "{label}", 1200);
        }}
      }});
    </script>
    """
    components.html(html, height=44)


def scroll_to_anchor(anchor_id: str) -> None:
    html = f"""
    <script>
      const el = window.parent.document.getElementById("{anchor_id}");
      if (el) {{
        el.scrollIntoView({{behavior: "smooth", block: "start"}});
      }}
    </script>
    """
    components.html(html, height=0)


def resolve_pdf_path(
    *,
    project_root: Path,
    data_raw_dir: Path,
    metadata: dict,
) -> Optional[Path]:
    """
    Cari path PDF dari metadata (file_path/source) atau Fallback: data_raw_dir/source_file.
    """
    # kandidat dari metadata
    for k in ["file_path", "source"]:
        v = metadata.get(k)
        if isinstance(v, str) and v.strip():
            p = Path(v)
            if not p.is_absolute():
                p = (project_root / p).resolve()
            if p.exists():
                return p

    # fallback dari source_file
    sf = metadata.get("source_file")
    if isinstance(sf, str) and sf.strip():
        p2 = (project_root / data_raw_dir / sf).resolve()
        if p2.exists():
            return p2

    return None


def render_pdf_viewer(
    pdf_path: Path,
    page: int = 1,
    height: int = 900,
    *,
    highlight_terms: Optional[Sequence[str]] = None,
    zoom: float = 2.0,
    max_hits_per_term: int = 50,
) -> None:
    """
    Render PDF per halaman jadi gambar (PNG) agar tidak diblok Chrome.
    + Optional highlight berdasarkan keyword (fitz.search_for).

    - page: 1-indexed
    - highlight_terms: list kata/frasas untuk di-highlight (case-insensitive by default)
    - zoom: 2.0 = lebih tajam

    Menggunakan PyMuPDF: `pymupdf` (import `fitz`).
    """
    try:
        import fitz  # PyMuPDF
    except Exception:
        st.warning("PyMuPDF (fitz) belum tersedia. Install: pip install pymupdf")
        return

    if not pdf_path.exists():
        st.warning("File PDF tidak ditemukan.")
        return

    # --- normalize terms ---
    terms: list[str] = []
    if highlight_terms:
        seen = set()
        for t in highlight_terms:
            if not isinstance(t, str):
                continue
            tt = t.strip()
            if not tt:
                continue
            key = tt.lower()
            if key in seen:
                continue
            seen.add(key)
            terms.append(tt)
    # batasi jumlah terms agar tidak berat
    terms = terms[:8]

    # cache render base image per (file, page, zoom)
    @st.cache_data(show_spinner=False)
    def _render_page_png(path_str: str, page_num: int, z: float) -> bytes:
        doc = fitz.open(path_str)
        try:
            p = max(1, min(int(page_num), doc.page_count))
            page_obj = doc.load_page(p - 1)
            mat = fitz.Matrix(z, z)
            pix = page_obj.get_pixmap(matrix=mat, alpha=False)
            return pix.tobytes("png")
        finally:
            doc.close()

    png_bytes = _render_page_png(str(pdf_path), int(page), float(zoom))

    # jika tidak ada terms -> tampilkan langsung
    if not terms:
        try:
            st.image(png_bytes, caption=f"{pdf_path.name} — page {page}", width="stretch")
        except TypeError:
            st.image(png_bytes, caption=f"{pdf_path.name} — page {page}", use_container_width=True)
        return

    # --- cari rectangles keyword di page (fitz) ---
    doc2 = fitz.open(str(pdf_path))
    try:
        p2 = max(1, min(int(page), doc2.page_count))
        page_obj2 = doc2.load_page(p2 - 1)

        rects_all: list[fitz.Rect] = []
        total_hits = 0
        for term in terms:
            try:
                rects = page_obj2.search_for(term, hit_max=max_hits_per_term)
            except TypeError:
                # kompatibilitas versi lama fitz (tanpa hit_max)
                rects = page_obj2.search_for(term)
                rects = rects[:max_hits_per_term]
            rects_all.extend(rects)
            total_hits += len(rects)

        # kalau tidak ada match, tampilkan tanpa highlight tapi beri info
        if total_hits == 0:
            st.caption(f"Highlight: {', '.join(terms)} — tidak ditemukan pada page ini.")
            try:
                st.image(png_bytes, caption=f"{pdf_path.name} — page {page}", width="stretch")
            except TypeError:
                st.image(
                    png_bytes, caption=f"{pdf_path.name} — page {page}", use_container_width=True
                )
            return

        # --- gambar highlight di atas image (butuh Pillow) ---
        try:
            from PIL import Image, ImageDraw  # type: ignore
        except Exception:
            st.caption(
                "Highlight term ditemukan, tapi Pillow belum terpasang. "
                "Install: pip install pillow (atau tambahkan ke requirements.txt)."
            )
            try:
                st.image(png_bytes, caption=f"{pdf_path.name} — page {page}", width="stretch")
            except TypeError:
                st.image(
                    png_bytes, caption=f"{pdf_path.name} — page {page}", use_container_width=True
                )
            return

        import io as _io

        img = Image.open(_io.BytesIO(png_bytes)).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        z = float(zoom)
        # warna kuning transparan + outline
        fill = (255, 255, 0, 80)
        outline = (255, 200, 0, 160)

        # batasi total rect agar tidak berat
        rects_all = rects_all[:300]

        for r in rects_all:
            x0 = int(r.x0 * z)
            y0 = int(r.y0 * z)
            x1 = int(r.x1 * z)
            y1 = int(r.y1 * z)

            # margin tipis
            pad = 2
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(img.size[0] - 1, x1 + pad)
            y1 = min(img.size[1] - 1, y1 + pad)

            draw.rectangle([x0, y0, x1, y1], fill=fill, outline=outline, width=2)

        composed = Image.alpha_composite(img, overlay).convert("RGB")

        st.caption(f"Highlight: {', '.join(terms)} — {total_hits} match pada page ini.")
        try:
            st.image(composed, caption=f"{pdf_path.name} — page {page}", width="stretch")
        except TypeError:
            st.image(composed, caption=f"{pdf_path.name} — page {page}", use_container_width=True)

    finally:
        doc2.close()
