"""RunPod serverless handler for DeepSeek-OCR-2."""

import base64
import io
import os
import tempfile
import urllib.request

import fitz  # PyMuPDF – used for PDF → image conversion
import runpod
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# Model loading – happens once when the container starts (warm-up).
# Set MODEL_NAME to override the default HuggingFace model ID, or point it at
# a local directory / network-volume path that already contains the weights.
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-ai/DeepSeek-OCR-2")

print(f"Loading tokenizer from {MODEL_NAME} …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

print(f"Loading model from {MODEL_NAME} …")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    _attn_implementation="flash_attention_2",
    trust_remote_code=True,
    use_safetensors=True,
)
model = model.eval().cuda().to(torch.bfloat16)
print("Model ready.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_PROMPTS = {
    "document": "<image>\n<|grounding|>Convert the document to markdown. ",
    "ocr": "<image>\nFree OCR. ",
}

# PDF rendering resolution (matches upstream vLLM config: dpi=144)
PDF_DPI = 144


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fetch_bytes(source: str) -> bytes:
    """Return raw bytes from a URL or a base64-encoded string."""
    if source.startswith("http://") or source.startswith("https://"):
        with urllib.request.urlopen(source) as resp:
            return resp.read()
    # Strip optional data-URI prefix, e.g. "data:image/jpeg;base64,..."
    if "," in source:
        source = source.split(",", 1)[1]
    return base64.b64decode(source)


def _image_extension(data: bytes) -> str:
    """Return a file extension that matches the image magic bytes."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:2] == b"\xff\xd8":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".png"  # safe fallback – Pillow and the model accept PNG for unknowns
    return data[:4] == b"%PDF"


def _pdf_to_images(pdf_bytes: bytes, dpi: int = PDF_DPI) -> list:
    """Convert every page of a PDF to a PIL Image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    images = []
    for page in doc:
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.open(io.BytesIO(pixmap.tobytes("png")))
        images.append(img)
    doc.close()
    return images


def _run_ocr(image_path: str, prompt: str, base_size: int, image_size: int, crop_mode: bool, output_dir: str) -> str:
    """Run model inference on a single image file and return the text result."""
    return model.infer(
        tokenizer,
        prompt=prompt,
        image_file=image_path,
        output_path=output_dir,
        base_size=base_size,
        image_size=image_size,
        crop_mode=crop_mode,
        save_results=False,
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def handler(job):
    """Process an OCR job.

    Input fields
    ------------
    image       : required – URL *or* base64-encoded image or PDF
                  (JPEG/PNG/PDF/etc.)
    prompt      : optional – raw prompt string, or one of the shorthand keys
                  ``"document"`` (default) / ``"ocr"``
    base_size   : optional – int, default 1024
    image_size  : optional – int, default 768
    crop_mode   : optional – bool, default True

    Output
    ------
    Single image  → ``{"result": "<text>"}``
    PDF           → ``{"pages": ["<page 1 text>", "<page 2 text>", …]}``
    """
    job_input = job["input"]

    image_data = job_input.get("image")
    if not image_data:
        return {"error": "No image provided. Supply 'image' as a URL or base64 string."}

    # Resolve prompt
    raw_prompt = job_input.get("prompt", "document")
    prompt = SUPPORTED_PROMPTS.get(raw_prompt, raw_prompt)

    try:
        base_size = int(job_input.get("base_size", 1024))
    except (ValueError, TypeError):
        return {"error": "Invalid base_size: must be an integer."}

    try:
        image_size = int(job_input.get("image_size", 768))
    except (ValueError, TypeError):
        return {"error": "Invalid image_size: must be an integer."}

    crop_mode = bool(job_input.get("crop_mode", True))

    # Fetch raw bytes (works for both URLs and base64)
    try:
        raw_bytes = _fetch_bytes(image_data)
    except Exception as exc:
        return {"error": f"Failed to load image: {exc}"}

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        try:
            if _is_pdf(raw_bytes):
                # PDF path: render each page to an image, OCR each one
                pil_images = _pdf_to_images(raw_bytes)
                page_results = []
                for idx, pil_img in enumerate(pil_images):
                    page_path = os.path.join(tmpdir, f"page_{idx:04d}.png")
                    pil_img.save(page_path, format="PNG")
                    page_results.append(
                        _run_ocr(page_path, prompt, base_size, image_size, crop_mode, output_dir)
                    )
                return {"pages": page_results}
            else:
                # Single-image path
                ext = _image_extension(raw_bytes)
                image_path = os.path.join(tmpdir, f"input_image{ext}")
                with open(image_path, "wb") as fh:
                    fh.write(raw_bytes)
                result = _run_ocr(image_path, prompt, base_size, image_size, crop_mode, output_dir)
                return {"result": result}

        except Exception as exc:
            return {"error": f"Inference failed: {exc}"}


runpod.serverless.start({"handler": handler})
