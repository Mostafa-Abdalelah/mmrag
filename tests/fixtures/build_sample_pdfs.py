from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas


def build_sample_pdf(path: Path, n_pages: int, tag: str) -> None:
    c = canvas.Canvas(str(path), pagesize=LETTER)
    for i in range(n_pages):
        c.setFont("Helvetica", 24)
        c.drawString(72, 720, f"{tag} - page {i + 1}")
        c.drawString(72, 680, f"sample body line for page {i + 1}")
        c.showPage()
    c.save()


def ensure_sample_pdfs(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "one": out_dir / "sample_1page.pdf",
        "three": out_dir / "sample_3page.pdf",
    }
    if not paths["one"].exists():
        build_sample_pdf(paths["one"], 1, "ONE")
    if not paths["three"].exists():
        build_sample_pdf(paths["three"], 3, "THREE")
    return paths
