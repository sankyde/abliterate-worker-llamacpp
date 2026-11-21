FROM ghcr.io/ggml-org/llama.cpp:server-cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && (pip3 install --no-cache-dir --break-system-packages \
            runpod==1.7.13 httpx==0.28.1 huggingface-hub==0.30.2 hf-transfer==0.1.9 \
        || pip3 install --no-cache-dir \
            runpod==1.7.13 httpx==0.28.1 huggingface-hub==0.30.2 hf-transfer==0.1.9)

ENV HF_HUB_ENABLE_HF_TRANSFER=1
COPY handler.py /app/handler.py

ENTRYPOINT ["python3", "-u", "/app/handler.py"]
