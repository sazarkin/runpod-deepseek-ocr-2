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
# Uncomment the two lines below and set MODEL_NAME to the HuggingFace model ID
# or local path. Leave commented to download at container start-up instead.
# ARG MODEL_NAME=deepseek-ai/DeepSeek-OCR-2
# RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoTokenizer.from_pretrained('${MODEL_NAME}', trust_remote_code=True); AutoModel.from_pretrained('${MODEL_NAME}', trust_remote_code=True, use_safetensors=True)"

# Add handler
ADD handler.py .

# Run the handler
CMD python -u /handler.py

