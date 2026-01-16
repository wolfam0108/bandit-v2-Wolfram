> ### Please consider giving back to the community if you have benefited from this work.
>
> If you've **benefited commercially from this work**, which we've poured significant effort into and released under permissive licenses, we hope you've found it valuable! While these licenses give you lots of freedom, we believe in nurturing a vibrant ecosystem where innovation can continue to flourish.
>
> So, as a gesture of appreciation and responsibility, we strongly urge commercial entities that have gained from this software to consider making voluntary contributions to music-related non-profit organizations of your choice. Your contribution directly helps support the foundational work that empowers your commercial success and ensures open-source innovation keeps moving forward.
>
> Some suggestions for the beneficiaries are provided [here](https://github.com/the-secret-source/nonprofits). Please do not hesitate to contribute to the list by opening pull requests there.

---

# Bandit Model (Re)implementation

Bandit model (re)implementation as seen in "Remastering Divide and Remaster: A Cinematic Audio Source Separation Dataset with Multilingual Support" by Karn N. Watcharasupat, Chih-Wei Wu, and Iroro Orife. [[arXiv]](https://arxiv.org/abs/2407.07275).

Model weights are available on Zenodo [here](https://zenodo.org/records/12701995).

PS: I'm hoping to get this repo to eventually unify all [Bandit-based models](https://github.com/kwatcharasupat/source-separation-landing), but research code gets refactored incompatibly a lot so let's see where this goes.

---

## Fork Changes (Wolfram)

This fork adds **`simple_inference.py`** — a standalone inference script that works without Ray, Netflix internal packages, or complex dependencies.

### What's Added

- **`simple_inference.py`** — Simple chunked inference script for local GPU
- Processes audio of **any length** (chunked with crossfade)
- **~17x realtime** on RTX 4090, ~2GB VRAM usage
- Preserves **stereo** (processes each channel separately)
- No modifications to original source code

### Why This Fork?

The original repo has dependencies on:
- Netflix internal packages (`nflx-*`, `jasper`, `metatron`)
- Ray distributed computing framework
- `asteroid` package (requires C++ compilation of `pesq`)

These make direct usage impossible for most users. This fork provides a minimal working solution.

---

## Quick Start

### 1. Create Environment

```bash
conda create -n bandit-v2 python=3.10 -y
conda activate bandit-v2
```

### 2. Install PyTorch with CUDA 11.8

> **Note:** Original `torch==2.0.0` is no longer available. Use 2.5.0+

```bash
pip install torch==2.5.0+cu118 torchaudio==2.5.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Dependencies

> **Note:** We skip `asteroid` package as it requires C++ build tools for `pesq`. It's not needed for inference.

```bash
pip install "pytorch_lightning>=2.3.0" hydra-core omegaconf librosa soundfile einops tqdm julius huggingface-hub pyyaml scipy pandas
```

### 4. Download Model Weights

Download `checkpoint-multi.ckpt` from [Zenodo](https://zenodo.org/records/12701995).

### 5. Run Inference

```bash
python simple_inference.py \
    --checkpoint path/to/checkpoint-multi.ckpt \
    --audio path/to/audio.wav \
    --output path/to/output_folder
```

---

## Usage

### Basic

```bash
python simple_inference.py -c checkpoint.ckpt -a audio.wav
```

### Full Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `-c, --checkpoint` | (required) | Path to .ckpt file |
| `-a, --audio` | (required) | Input audio file |
| `-o, --output` | ./estimates/ | Output directory |
| `--chunk` | 30.0 | Chunk size in seconds |
| `--overlap` | 2.0 | Overlap for crossfade |
| `--stems` | speech music sfx | Stems to separate |
| `--no-half` | False | Disable mixed precision |

### Output Files

```
output_folder/
├── speech_estimate.wav  # Dialogue/speech
├── music_estimate.wav   # Music
└── sfx_estimate.wav     # Sound effects
```

---

## Performance

Tested on RTX 4090 with 7-minute stereo audio @ 48kHz:

| Metric | Value |
|--------|-------|
| Speed | **16.9x realtime** |
| Time | 24.9 seconds |
| Peak VRAM | 1.99 GB |
| Chunk size | 30s |
| Overlap | 2s |

---

## How It Works

1. **Chunked Processing** — Audio is split into 30-second chunks with 2-second overlap
2. **Per-Channel Stereo** — Each channel processed separately (model is mono)
3. **Crossfade Blending** — Overlapping regions are blended with linear fade
4. **Memory Efficient** — Only one chunk in VRAM at a time, results moved to CPU immediately
5. **Mixed Precision** — Uses `torch.cuda.amp.autocast()` for speed

---

## Technical Details

Model architecture:
- 64 frequency bands (musical band-split)
- 8 dual-path RNN modules (GRU, bidirectional)
- Embedding dim: 128, RNN dim: 256
- STFT: n_fft=2048, hop=512
- Sample rate: 48000 Hz

---

## License

Apache 2.0 (same as original)

---

## Credits

- Original Bandit model: [kwatcharasupat/bandit-v2](https://github.com/kwatcharasupat/bandit-v2)
- Paper: [arXiv:2407.07275](https://arxiv.org/abs/2407.07275)
- `simple_inference.py`: Added by Wolfram for standalone inference
