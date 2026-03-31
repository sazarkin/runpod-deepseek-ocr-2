"""RunPod serverless handler for DeepSeek-OCR-2."""

import base64
import os
import tempfile
import urllib.request

import runpod
import torch
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
# Helper
# ---------------------------------------------------------------------------
SUPPORTED_PROMPTS = {
    "document": "<image>\n<|grounding|>Convert the document to markdown. ",
    "ocr": "<image>\nFree OCR. ",
}


def _save_image(image_data: str, dest_path: str) -> None:
    """Write image bytes to *dest_path* from a URL or a base64-encoded string."""
    if image_data.startswith("http://") or image_data.startswith("https://"):
        urllib.request.urlretrieve(image_data, dest_path)
    else:
        # Strip optional data-URI prefix, e.g. "data:image/jpeg;base64,..."
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]
        image_bytes = base64.b64decode(image_data)
        with open(dest_path, "wb") as fh:
            fh.write(image_bytes)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def handler(job):
    """Process an OCR job.

    Input fields
    ------------
    image       : required – URL *or* base64-encoded image (JPEG/PNG/etc.)
    prompt      : optional – raw prompt string, or one of the shorthand keys
                  "document" (default) / "ocr"
    base_size   : optional – int, default 1024
    image_size  : optional – int, default 768
    crop_mode   : optional – bool, default True
    """
    job_input = job["input"]

    image_data = job_input.get("image")
    if not image_data:
        return {"error": "No image provided. Supply 'image' as a URL or base64 string."}

    # Resolve prompt
    raw_prompt = job_input.get("prompt", "document")
    prompt = SUPPORTED_PROMPTS.get(raw_prompt, raw_prompt)

    base_size = int(job_input.get("base_size", 1024))
    image_size = int(job_input.get("image_size", 768))
    crop_mode = bool(job_input.get("crop_mode", True))

    with tempfile.TemporaryDirectory() as tmpdir:
        image_path = os.path.join(tmpdir, "input_image.jpg")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(output_dir, exist_ok=True)

        try:
            _save_image(image_data, image_path)
        except Exception as exc:
            return {"error": f"Failed to load image: {exc}"}

        try:
            result = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=image_path,
                output_path=output_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=False,
            )
        except Exception as exc:
            return {"error": f"Inference failed: {exc}"}

    return {"result": result}


runpod.serverless.start({"handler": handler})
