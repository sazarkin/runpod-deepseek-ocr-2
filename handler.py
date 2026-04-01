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

# Compatibility shim: transformers ≥4.46 removed LlamaFlashAttention2, but the
# model's bundled modeling_deepseekv2.py still imports it unconditionally at
# module level. Alias it to LlamaAttention so the import succeeds; the actual
# attention implementation is still governed by _attn_implementation below.
import transformers.models.llama.modeling_llama as _llama_mod
if not hasattr(_llama_mod, "LlamaFlashAttention2"):
    _llama_mod.LlamaFlashAttention2 = _llama_mod.LlamaAttention

# ---------------------------------------------------------------------------
# Cached-model configuration.
# RunPod caches Hugging Face models at HF_CACHE_ROOT when a model ID is set
# on the endpoint.  Force offline mode so the worker never attempts a download.
# ---------------------------------------------------------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "deepseek-ai/deepseek-ocr-2")
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def resolve_snapshot_path(model_id: str) -> str:
    """Resolve the local snapshot path for a RunPod-cached Hugging Face model.

    RunPod stores the model under ``HF_CACHE_ROOT/models--<org>--<name>``.
    The directory name preserves the exact casing of the model ID configured
    on the endpoint, so the ``MODEL_NAME`` env-var must match that casing.

    Resolution order:
    1. Read every file under ``refs/`` and use the first hash whose snapshot
       directory exists (handles both ``refs/main`` and pinned-commit refs).
    2. Fall back to the lexicographically first subdirectory of ``snapshots/``.
    3. If the exact-case directory is missing, try a case-insensitive scan of
       ``HF_CACHE_ROOT`` so a casing mismatch produces a helpful error instead
       of a silent failure.
    """
    if "/" not in model_id:
        raise ValueError(f"model_id '{model_id}' must be in 'org/name' format")

    org, name = model_id.split("/", 1)
    dir_name = f"models--{org}--{name}"
    model_root = os.path.join(HF_CACHE_ROOT, dir_name)

    print(f"[ModelStore] MODEL_NAME: {model_id}")
    print(f"[ModelStore] Model root: {model_root}")

    # If the exact-case directory is missing, try a case-insensitive match so
    # we can surface a helpful error message showing what *is* available.
    # This scan only runs when the exact-case path doesn't exist (error path),
    # so there is no performance impact during normal startup.
    if not os.path.isdir(model_root) and os.path.isdir(HF_CACHE_ROOT):
        dir_name_lower = dir_name.lower()
        for entry in os.listdir(HF_CACHE_ROOT):
            if entry.lower() == dir_name_lower:
                found = os.path.join(HF_CACHE_ROOT, entry)
                print(
                    f"[ModelStore] WARNING: '{dir_name}' not found; "
                    f"using case-insensitive match '{entry}'. "
                    f"Set MODEL_NAME to match the endpoint model exactly."
                )
                model_root = found
                break
        else:
            available = [e for e in os.listdir(HF_CACHE_ROOT) if e.startswith("models--")]
            print(f"[ModelStore] Available model dirs: {available}")
            raise RuntimeError(
                f"Cached model not found: {model_id} "
                f"(looked for '{dir_name}' in {HF_CACHE_ROOT})"
            )

    snapshots_dir = os.path.join(model_root, "snapshots")
    refs_dir = os.path.join(model_root, "refs")

    # 1. Try every ref file (main, a pinned commit hash alias, etc.)
    if os.path.isdir(refs_dir):
        for ref_name in sorted(os.listdir(refs_dir)):
            ref_file = os.path.join(refs_dir, ref_name)
            if os.path.isfile(ref_file):
                with open(ref_file, "r") as f:
                    snapshot_hash = f.read().strip()
                candidate = os.path.join(snapshots_dir, snapshot_hash)
                if os.path.isdir(candidate):
                    print(f"[ModelStore] Using snapshot from refs/{ref_name}: {candidate}")
                    return candidate

    # 2. Fall back to the first snapshot directory (lexicographic order gives a
    #    deterministic result; snapshot dirs are content-hashes so there is no
    #    meaningful recency ordering without inspecting filesystem timestamps).
    if os.path.isdir(snapshots_dir):
        versions = [
            d for d in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, d))
        ]
        if versions:
            versions.sort()
            chosen = os.path.join(snapshots_dir, versions[0])
            print(f"[ModelStore] Using first available snapshot: {chosen}")
            return chosen

    raise RuntimeError(
        f"Cached model not found: {model_id} "
        f"(no snapshots under {model_root})"
    )


# ---------------------------------------------------------------------------
# Model loading – happens once when the container starts (warm-up).
# ---------------------------------------------------------------------------
LOCAL_PATH = resolve_snapshot_path(MODEL_NAME)

print(f"[ModelStore] Resolved local model path: {LOCAL_PATH}")
print(f"Loading tokenizer from {LOCAL_PATH} …")
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH, trust_remote_code=True, local_files_only=True)

print(f"Loading model from {LOCAL_PATH} …")
model = AutoModel.from_pretrained(
    LOCAL_PATH,
    _attn_implementation="flash_attention_2",
    trust_remote_code=True,
    use_safetensors=True,
    local_files_only=True,
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


def _is_pdf(data: bytes) -> bool:
    """Return True if the raw bytes look like a PDF."""
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
