# SGLang setup: failed host install (October 11), working container (October 12--13, 2026)

SGLang could not be started directly on the machine image used for vLLM. It
was then run successfully from its official container image, and the
measured SGLang results come from that container. This note records both, so
that the setup can be reproduced.

## Setup

| Item | Value |
| --- | --- |
| Instance | AWS g5.xlarge (1x NVIDIA A10G, 23 GB), us-east-2 |
| Image | Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.14 (Ubuntu 26.04), build 20261003 |
| Driver | 595.91.07 |
| CUDA toolkit | none installed system-wide (`/usr/local/cuda` absent); `nvidia-cuda-nvcc` available only as a pip package inside the Python environment |
| Python | 3.12.15 (via uv) |
| Model | meta-llama/Llama-3.1-8B-Instruct |

## What happened

Each step is a separate startup failure, fixed or worked around before the next appeared.

1. **DeepEP import requires a CUDA home.** `deep_ep/__init__.py` asserts a CUDA
   installation exists at import time (`assert cuda_home is not None`).
   Workaround: point `CUDA_HOME` at the pip-installed `nvidia/cu13` directory.
2. **Gated model access.** A new SSH session had no Hugging Face token set.
   Fixed by setting `HF_TOKEN` (an operator error, not an SGLang problem).
3. **Static memory fraction too low for a 24 GB GPU.** SGLang's default
   `--mem-fraction-static` (0.684 here) left no room for the KV cache after
   the 16 GB of weights loaded; it asked for more than 0.691. Raised to 0.85.
4. **Kernel compilation at startup.** With the above in place, a child process
   failed during startup with an empty `stderr` and SGLang shut itself down
   (exit -9 from its own process-tree kill, not the kernel OOM killer; `dmesg`
   showed no OOM event). This is consistent with a component compiling GPU
   code without a CUDA toolkit, the same root cause that required avoiding
   FlashInfer's sampler in vLLM.

Server flags at the last attempt:
`--context-length 4096 --attention-backend triton --sampling-backend pytorch --mem-fraction-static 0.85`

## Resolution: the official container (measured runs)

The same machine (same AMI, driver 595.91.07) ran SGLang from
`lmsysorg/sglang:v0.5.21` (digest
`sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469`),
which ships CUDA 13.0 and prebuilt kernels. The harness runs inside the same
container, so the process-group SIGKILL reaches the server, exactly as on the
host for vLLM. A one-trial smoke test passed first, and then
[`scripts/sgl_grid.sh`](../scripts/sgl_grid.sh) ran the 12-run grid and the
4-run radix-cache ablation (16 runs, 80 trials, none failed), on harness
commit `ddd8890`, whose `harness/` is identical to the vLLM runs' `90da903`.

Server flags: `--context-length 4096 --mem-fraction-static 0.85`, all else
default. The server logs record what the defaults selected on the A10G:
FlashInfer attention and sampling, CUDA graphs on, radix cache on (off in the
ablation via `--disable-radix-cache`), deterministic inference off.

## Original decision (October 11)

After about 30 minutes and four distinct failures, SGLang was deferred so that
the vLLM measurements could complete. The harness (`harness/`) and grid script
(`scripts/run_matrix.sh sglang`) already support SGLang unchanged.

## To retry on the host (not needed for the container route)

Use an image with a full CUDA toolkit matching the installed PyTorch (or
install one), then run `bash scripts/setup_instance.sh sglang`. The memory
fraction setting above will likely still be needed on a 24 GB GPU.
