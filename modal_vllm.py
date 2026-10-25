"""Serve one open-weight model on a Modal GPU behind an OpenAI-compatible API (vLLM).

Three steps, each from the notebook (or a terminal with the same env vars):
    modal run modal_vllm.py::preflight    # CPU only, ~1 cent: checks image, tokenizer, config
    modal deploy modal_vllm.py            # creates the endpoint (no GPU until first request)
    modal app stop collusion-m            # stop paying

Env vars read at deploy time: MODEL, APP_NAME, GPU, MAX_LEN, USE_HF_SECRET.
The endpoint requires the key in the Modal secret "vllm-api-key".
"""
import os
import subprocess
import threading

import modal

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-14B-Instruct")
APP_NAME = os.environ.get("APP_NAME", "collusion-m")
GPU = os.environ.get("GPU", "L40S")
MAX_LEN = os.environ.get("MAX_LEN", "8192")
USE_HF_SECRET = os.environ.get("USE_HF_SECRET", "1")

image = (
    # CUDA 12.8 devel base: matches the torch 2.8 cu128 build that vllm 0.11.0 installs.
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    # Pinned and checked together (24 Oct): unpinned vllm and transformers 5 both broke start-up.
    .pip_install("vllm==0.11.0", "transformers==4.57.1", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "MODEL": MODEL, "APP_NAME": APP_NAME,
          "GPU": GPU, "MAX_LEN": MAX_LEN, "USE_HF_SECRET": USE_HF_SECRET})
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
VOLUMES = {"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache}
SECRETS = [modal.Secret.from_name("vllm-api-key")] + (
    [modal.Secret.from_name("huggingface")] if USE_HF_SECRET == "1" else [])

app = modal.App(APP_NAME)


@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, timeout=30 * 60, cpu=4, memory=8192)
def preflight():
    """Runs on CPU (no GPU cost). Fails loudly on anything that would crash the GPU server."""
    import json

    import transformers
    import vllm
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer
    from vllm.transformers_utils.tokenizer import get_cached_tokenizer

    model, max_len = os.environ["MODEL"], int(os.environ["MAX_LEN"])
    print(f"[1/6] versions: vllm {vllm.__version__}, transformers {transformers.__version__}")

    print(f"[2/6] downloading weights for {model} into the cache volume (skips if cached)...")
    path = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt",
                                                    "*.model", "*.tiktoken"])
    hf_cache.commit()  # keep the download even though this container ends
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)
    print(f"      cached {size / 1e9:.1f} GB at {path}")

    cfg = AutoConfig.from_pretrained(path)
    archs = cfg.architectures or []
    print(f"[3/6] config: {archs}, context {cfg.max_position_embeddings}")
    assert max_len <= cfg.max_position_embeddings, "MAX_LEN is longer than the model supports"

    from vllm.model_executor.models import ModelRegistry
    supported = set(ModelRegistry.get_supported_archs())
    assert any(a in supported for a in archs), f"vLLM 0.11.0 does not support {archs}"
    print("[4/6] vLLM supports this architecture")

    tok = get_cached_tokenizer(AutoTokenizer.from_pretrained(path))
    print(f"[5/6] tokenizer loads in vLLM: {type(tok).__name__}")

    msgs = [{"role": "user", "content": "PRICE: 1.5"}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert prompt and "PRICE" in prompt
    print(f"[6/6] chat template works ({len(tok(prompt).input_ids)} tokens)")

    weights_gb = size / 1e9
    gpu_gb = {"L4": 24, "A10G": 24, "L40S": 48, "A100": 40, "A100-80GB": 80, "H100": 80}.get(
        os.environ["GPU"], 0)
    if gpu_gb:
        room = gpu_gb * 0.90 - weights_gb
        print(f"      memory: weights {weights_gb:.1f} GB on a {gpu_gb} GB {os.environ['GPU']}"
              f" leaves ~{room:.1f} GB for the KV cache")
        assert room > 2, "Model too big for this GPU: pick a bigger GPU or the AWQ model"
    print("PREFLIGHT PASSED " + json.dumps({"model": model, "gpu": os.environ["GPU"]}))


@app.function(
    image=image,
    gpu=GPU,
    scaledown_window=10 * 60,       # shut the GPU down after 10 idle minutes
    timeout=12 * 60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
    max_containers=1,
)
@modal.concurrent(max_inputs=256)
@modal.web_server(port=8000, startup_timeout=15 * 60)
def serve():
    cmd = [
        "vllm", "serve", os.environ["MODEL"],
        "--host", "0.0.0.0", "--port", "8000",
        "--served-model-name", os.environ["MODEL"],
        "--max-model-len", os.environ["MAX_LEN"],
        "--gpu-memory-utilization", "0.90",
        "--enable-prefix-caching",
        "--enforce-eager",          # skip torch.compile and CUDA graphs: faster, safer start
        "--api-key", os.environ["VLLM_API_KEY"],
    ]
    proc = subprocess.Popen(cmd)

    def stop_if_vllm_dies():
        # Without this, a crashed vLLM leaves the GPU waiting (and billing) until the timeout.
        code = proc.wait()
        print(f"vLLM exited with code {code}; shutting this container down now.", flush=True)
        os._exit(1)

    threading.Thread(target=stop_if_vllm_dies, daemon=True).start()
