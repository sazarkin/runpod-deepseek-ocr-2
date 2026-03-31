# RunPod · DeepSeek-OCR-2 Serverless Worker

A RunPod serverless worker that exposes [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) as an API endpoint.  
The model converts document images to Markdown or plain text using a 7 B vision-language architecture.

---

## API

### Input

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `image` | string | ✅ | — | Public image/PDF URL **or** base64-encoded image or PDF (optionally with a `data:…;base64,` prefix) |
| `prompt` | string | | `"document"` | Shorthand `"document"` (Markdown output) or `"ocr"` (plain text), or a raw prompt string |
| `base_size` | int | | `1024` | Base tile size for the encoder |
| `image_size` | int | | `768` | Resolution of each crop tile |
| `crop_mode` | bool | | `true` | Enable dynamic multi-crop resolution |

### Output

```json
{ "result": "<extracted text or markdown>" }
```

For PDF input, one result per page is returned:

```json
{ "pages": ["<page 1 text>", "<page 2 text>", "…"] }
```

On error:

```json
{ "error": "<error message>" }
```

---

## Prompts

| Shorthand | Use case |
|---|---|
| `document` | Documents with layouts, tables, formulas — output is Markdown |
| `ocr` | Plain text extraction without layout |

Full prompts used internally:

```
document → <image>\n<|grounding|>Convert the document to markdown. 
ocr      → <image>\nFree OCR. 
```

You can also pass any custom prompt string in the `prompt` field.

---

## Example Requests

### Python (recommended)

```python
import runpod
import base64

client = runpod.api.RunPodClient(api_key="YOUR_API_KEY")

# --- Image from URL ---
response = client.run_sync(
    endpoint_id="YOUR_ENDPOINT_ID",
    job_input={"image": "https://example.com/invoice.png", "prompt": "document"},
)
print(response["output"]["result"])

# --- Image from file (base64) ---
with open("scan.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = client.run_sync(
    endpoint_id="YOUR_ENDPOINT_ID",
    job_input={"image": img_b64, "prompt": "document"},
)
print(response["output"]["result"])

# --- PDF from file (base64) — one result per page ---
with open("document.pdf", "rb") as f:
    pdf_b64 = base64.b64encode(f.read()).decode()

response = client.run_sync(
    endpoint_id="YOUR_ENDPOINT_ID",
    job_input={"image": pdf_b64, "prompt": "document"},
)
for i, page_text in enumerate(response["output"]["pages"], 1):
    print(f"--- Page {i} ---\n{page_text}")
```

### JSON (RunPod Playground / curl)

```json
{ "input": { "image": "https://example.com/invoice.png", "prompt": "document" } }
```

```json
{ "input": { "image": "<base64-encoded PDF>", "prompt": "document" } }
```

---

## Deployment

### Requirements

- CUDA 11.8 (tested; CUDA 12.x also supported)
- ≥ 24 GB VRAM (RTX 3090/4090, A6000, A100, H100, …)
- ≥ 50 GB container disk (for model weights)

### 1. GitHub Integration (recommended)

Connect this repository to RunPod Serverless.  
RunPod will automatically build and deploy the worker on every push.  
See the [RunPod Deploy with GitHub Guide](https://docs.runpod.io/serverless/github-integration).

### 2. Manual Docker Build

```bash
docker build -t your-registry/deepseek-ocr-2 .
docker push your-registry/deepseek-ocr-2
```

Then create a new Template / Endpoint in the RunPod UI pointing at your image.

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `MODEL_NAME` | HuggingFace model ID or local path to the weights | `deepseek-ai/DeepSeek-OCR-2` |

Set `MODEL_NAME` to a network-volume path (e.g. `/runpod-volume/DeepSeek-OCR-2`) to avoid downloading weights on every cold start.

---

## Local Testing

```bash
pip install -r requirements.txt
# flash-attn must be installed separately
pip install flash-attn==2.7.3 --no-build-isolation

python handler.py
```

Modify `test_input.json` with your image URL before running.

---

## Further Information

- [DeepSeek-OCR-2 on HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)
- [DeepSeek-OCR-2 GitHub](https://github.com/deepseek-ai/DeepSeek-OCR-2)
- [RunPod Serverless Documentation](https://docs.runpod.io/serverless/overview)
- [RunPod Python SDK](https://github.com/runpod/runpod-python)

