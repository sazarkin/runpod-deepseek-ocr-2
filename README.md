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

| Shorthand | Full prompt | Use case |
|---|---|---|
| `document` | `<image>\n<|grounding|>Convert the document to markdown. ` | Documents with layouts, tables, formulas |
| `ocr` | `<image>\nFree OCR. ` | Plain text extraction without layout |

You can also pass any custom prompt string in the `prompt` field.

---

## Example Request

```json
{
  "input": {
    "image": "https://example.com/invoice.png",
    "prompt": "document"
  }
}
```

```json
{
  "input": {
    "image": "https://example.com/document.pdf",
    "prompt": "document"
  }
}
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

