FROM runpod/base:0.6.3-cuda11.8.0

# Set python3.11 as the default python
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

# Install dependencies
COPY requirements.txt /requirements.txt
RUN uv pip install --upgrade -r /requirements.txt --no-cache-dir --system

# Install flash-attn (requires --no-build-isolation; pinned for reproducibility)
RUN uv pip install flash-attn==2.7.3 --no-build-isolation --no-cache-dir --system

# Optional: pre-download the model into the image to avoid cold-start latency.
# Uncomment and set --build-arg MODEL_NAME=... when building:
#   docker build --build-arg MODEL_NAME=deepseek-ai/DeepSeek-OCR-2 .
# ARG MODEL_NAME=deepseek-ai/DeepSeek-OCR-2
# RUN python -c "import os; from transformers import AutoModel, AutoTokenizer; n=os.environ.get('MODEL_NAME','deepseek-ai/DeepSeek-OCR-2'); AutoTokenizer.from_pretrained(n, trust_remote_code=True); AutoModel.from_pretrained(n, trust_remote_code=True, use_safetensors=True)"

# Add handler
ADD handler.py .

# Run the handler
CMD python -u /handler.py

