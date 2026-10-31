# Acoustic Physics and Auditory Intelligence Labs

A collection of six Python labs covering audio analysis, room acoustics, wave propagation, and neural networks. 

The labs run on a CPU, with optional NVIDIA CUDA or Huawei Ascend 910C acceleration for the PyTorch experiments.

## What's inside

| Lab | Topic | What to explore |
| --- | --- | --- |
| 01 | Signals and spectra | Sampling, FFT, STFT, log-Mel features, and aliasing |
| 02 | Room acoustics and denoising | Synthetic room responses and changes in SNR |
| 03 | CNN classification | Spectrogram-based classification under shifted conditions |
| 04 | Transformer attention | Self-attention, normalization, and classification accuracy |
| 05 | Wave propagation (FDTD) | Grid convergence and CFL stability in one dimension |
| 06 | DeepONet for waves | Operator learning and generalization to unseen modes |

Labs 01, 02, and 05 use NumPy/SciPy and always run on the CPU. Labs 03, 04, and 06 use PyTorch and can run on a CPU, CUDA GPU, or Ascend NPU.

## Getting started

Python 3.11 or 3.12 is recommended. From the repository root, create an environment and install the common dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it with `.venv\Scripts\activate` (Command Prompt) or `.venv\Scripts\Activate.ps1` (PowerShell).

Install the PyTorch build for your hardware, then run the labs.

### CPU

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python cpu/run.py --all --report outputs/cpu-report.json
```

A CPU-only machine is enough to complete all six labs.

### NVIDIA GPU

Check your driver with `nvidia-smi`, then install a compatible CUDA build using the [PyTorch installation guide](https://pytorch.org/get-started/locally/). For example, if CUDA 12.8 is recommended:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python gpu/run.py --probe --strict-device
python gpu/run.py --all --strict-device --report outputs/gpu-report.json
```

To allow a CPU fallback when the GPU fails its initial check, leave out `--strict-device`.

### Huawei Ascend 910C

Use a Linux environment with the Ascend driver, firmware, and CANN packages installed. Match your PyTorch and `torch-npu` versions to the installed CANN release using the [Ascend compatibility matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md).

Activate CANN and check the NPU before running:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info
bash npu/run.sh --probe --strict-device
bash npu/run.sh --all --strict-device --report outputs/npu-report.json
```

If CANN is installed elsewhere, point the launcher to its environment script:

```bash
CANN_ENV_SCRIPT=/path/to/set_env.sh bash npu/run.sh --all --strict-device
```

The launcher can fall back to CPU if the initial NPU check fails and `--strict-device` is omitted. Once training has started, a device error is reported rather than silently restarting on CPU. Keep `--strict-device` when you need to verify execution on the requested accelerator.

## Running individual labs

```bash
python cpu/run.py                          # Labs 01, 02, and 05
python gpu/run.py --lab 03_cnn_classification
python npu/run.py --lab 06_deeponet_wave
python verify.py --all --backend auto
```

The automatic backend check tries CUDA, then NPU, then CPU. Each lab saves a PNG figure and JSON metrics under `outputs/`; Labs 01 and 02 also save WAV audio. Generated outputs are excluded from Git.

## Repository layout

| Path | Contents |
| --- | --- |
| `labs/` | Six experiment implementations |
| `cpu/`, `gpu/`, `npu/` | Platform-specific launchers |
| `acoustic_labs/` | Device checks and result handling |
| `tests/` | Tests and validation records |
| `verify.py` | End-to-end checks |

The experiments share the same implementations across platforms, making it easier to compare results and extend the code for coursework or independent study.
