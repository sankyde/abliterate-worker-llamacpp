"""Runpod Serverless queue worker wrapping llama-server with the exact worker-v1-vllm OpenAI queue contract.

Accepts {"openai_route": "/v1/chat/completions" | "/v1/completions", "openai_input": {...}} and forwards to a local
llama-server process with continuous batching, flash attention, KV prefix caching, and Jinja tool/reasoning support.
"""

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx
import runpod

HOST = "127.0.0.1"
PORT = int(os.environ.get("LLAMA_PORT", "8080"))
BASE_URL = f"http://{HOST}:{PORT}"
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "8"))
SERVED_MODEL = os.environ.get("OPENAI_SERVED_MODEL_NAME_OVERRIDE") or os.environ.get("MODEL_NAME", "")
ALLOWED_ROUTES = {"/v1/chat/completions", "/v1/completions"}


def find_llama_server() -> str:
    for candidate in ("/app/llama-server", "/llama-server", "/server", shutil.which("llama-server")):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("llama-server binary not found in container")


def find_cached_gguf(repo_id: str, filename: str) -> Path | None:
    """Locate a GGUF file in Runpod's Model Store cache or HF cache without hitting the network."""
    slug = f"models--{repo_id.replace('/', '--')}"
    roots = [
        Path("/runpod-volume/huggingface-cache/hub"),
        Path("/runpod-volume/huggingface-cache"),
        Path(os.environ.get("HF_HOME", "/root/.cache/huggingface")) / "hub",
        Path("/root/.cache/huggingface/hub"),
    ]
    for root in roots:
        repo_dir = root / slug
        if repo_dir.is_dir():
            for match in repo_dir.rglob(filename):
                if match.is_file() or match.is_symlink():
                    return match
        direct = root / filename
        if direct.is_file():
            return direct
    return None


def resolve_model_path() -> str:
    repo_id = os.environ["MODEL_NAME"]
    gguf_file = os.environ.get("GGUF_FILE", "").strip()
    if not gguf_file:
        raise RuntimeError("GGUF_FILE env var is required for worker-llamacpp")
    cached = find_cached_gguf(repo_id, gguf_file)
    if cached is not None:
        print(f"[worker-llamacpp] using cached GGUF: {cached}", flush=True)
        return str(cached)
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        # Check if Model Store placed files anywhere under /runpod-volume
        rp_root = Path("/runpod-volume")
        if rp_root.is_dir():
            for match in rp_root.rglob(gguf_file):
                print(f"[worker-llamacpp] found GGUF under /runpod-volume: {match}", flush=True)
                return str(match)
        raise RuntimeError(f"HF_HUB_OFFLINE=1 and {gguf_file} not found in cache for {repo_id}")
    from huggingface_hub import hf_hub_download

    print(f"[worker-llamacpp] downloading {repo_id}/{gguf_file} ...", flush=True)
    path = hf_hub_download(repo_id=repo_id, filename=gguf_file, token=os.environ.get("HF_TOKEN") or None)
    print(f"[worker-llamacpp] downloaded to {path}", flush=True)
    return path


def start_llama_server(model_path: str) -> subprocess.Popen:
    binary = find_llama_server()
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "32768"))
    total_ctx = int(os.environ.get("LLAMA_CTX_SIZE", str(max_model_len * min(MAX_CONCURRENCY, 4))))
    cmd = [
        binary,
        "-m", model_path,
        "--host", HOST,
        "--port", str(PORT),
        "-ngl", os.environ.get("LLAMA_NGL", "99"),
        "--ctx-size", str(total_ctx),
        "--parallel", str(MAX_CONCURRENCY),
        "--cont-batching",
        "--jinja",
        "--no-webui",
    ]
    if SERVED_MODEL:
        cmd.extend(["--alias", SERVED_MODEL])
    extra = os.environ.get("LLAMA_EXTRA_ARGS", "").strip()
    if extra:
        cmd.extend(shlex.split(extra))
    print(f"[worker-llamacpp] launching: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    deadline = time.monotonic() + int(os.environ.get("LLAMA_STARTUP_TIMEOUT", "600"))
    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early with code {proc.returncode}")
            try:
                r = client.get(f"{BASE_URL}/health")
                if r.status_code == 200:
                    print("[worker-llamacpp] llama-server is ready", flush=True)
                    return proc
            except httpx.HTTPError:
                pass
            time.sleep(1.0)
    proc.terminate()
    raise RuntimeError("llama-server did not become healthy before timeout")


def normalize_payload(openai_input: dict) -> dict:
    payload = dict(openai_input)
    # Strip vLLM-only fields that llama-server rejects or ignores.
    payload.pop("cache_salt", None)
    payload.setdefault("cache_prompt", True)
    if SERVED_MODEL:
        payload["model"] = SERVED_MODEL
    # Ensure llama-server returns usage on streaming requests when requested.
    if payload.get("stream") is True:
        opts = dict(payload.get("stream_options") or {})
        opts["include_usage"] = True
        payload["stream_options"] = opts
    return payload


def patch_response_obj(obj: dict) -> dict:
    """Align llama-server's response with vLLM's OpenAI format (served model name + cached_tokens)."""
    if SERVED_MODEL and "model" in obj:
        obj["model"] = SERVED_MODEL
    timings = obj.pop("timings", None)
    usage = obj.get("usage")
    if isinstance(usage, dict):
        cached = None
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
            cached = details["cached_tokens"]
        elif isinstance(timings, dict) and isinstance(timings.get("cache_n"), int):
            cached = timings["cache_n"]
        if cached is not None:
            usage["prompt_tokens_details"] = {**(details if isinstance(details, dict) else {}), "cached_tokens": cached}
    return obj


HTTP_CLIENT: httpx.AsyncClient | None = None


def get_http() -> httpx.AsyncClient:
    global HTTP_CLIENT
    if HTTP_CLIENT is None:
        HTTP_CLIENT = httpx.AsyncClient(
            base_url=BASE_URL,
            timeout=httpx.Timeout(600.0, connect=10.0),
            limits=httpx.Limits(max_connections=MAX_CONCURRENCY * 2, max_keepalive_connections=MAX_CONCURRENCY),
        )
    return HTTP_CLIENT


async def handler(job: dict):
    job_input = job.get("input") or {}
    route = job_input.get("openai_route")
    openai_input = job_input.get("openai_input")
    if route not in ALLOWED_ROUTES or not isinstance(openai_input, dict):
        yield {"error": f"vLLM returned HTTP 400: unsupported route {route!r}"}
        return

    payload = normalize_payload(openai_input)
    stream = payload.get("stream") is True
    client = get_http()

    if not stream:
        resp = await client.post(route, json=payload)
        if resp.status_code >= 400:
            yield {"error": f"vLLM returned HTTP {resp.status_code}: {resp.text[:8000]}"}
            return
        yield patch_response_obj(resp.json())
        return

    async with client.stream("POST", route, json=payload) as resp:
        if resp.status_code >= 400:
            body = (await resp.aread()).decode("utf-8", "replace")[:8000]
            yield {"error": f"vLLM returned HTTP {resp.status_code}: {body}"}
            return
        async for line in resp.aiter_lines():
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                yield "data: [DONE]\n\n"
                continue
            try:
                obj = json.loads(data_str)
                if isinstance(obj, dict):
                    obj = patch_response_obj(obj)
                yield f"data: {json.dumps(obj)}\n\n"
            except ValueError:
                yield f"{line}\n\n"


if __name__ == "__main__":
    model_path = resolve_model_path()
    _server_proc = start_llama_server(model_path)
    runpod.serverless.start(
        {
            "handler": handler,
            "return_aggregate_stream": True,
            "concurrency_modifier": lambda _: MAX_CONCURRENCY,
        }
    )
