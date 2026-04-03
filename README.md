# SemDespeckSR-Semantic-Conditioned-SAR-Despeckling-and-Super-Resolution

**Joint SAR Despeckling and Super-Resolution via Wavelet-Conditioned ESRGAN with Semantic Scene Conditioning**

[![NISAR](https://img.shields.io/badge/Data-NISAR%20L--band-blue)](https://nisar.jpl.nasa.gov/)
[![SARLANG-1M](https://img.shields.io/badge/Conditioning-SARLANG--1M-green)](https://huggingface.co/datasets/YiminJimmy/SARLANG-1M)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red)](https://pytorch.org/)

## Overview

SAR-WaveSRNet is a joint despeckling and super-resolution framework for SAR (Synthetic Aperture Radar) imagery. It addresses the concurrent noise-resolution degradation problem by combining:

1. **Wavelet-domain preprocessing** — 2D DWT decomposes input into frequency subbands, giving the network explicit speckle-structure separation before learning
2. **Modified ESRGAN backbone** — Residual-in-Residual Dense Blocks (RRDB) for powerful feature extraction
3. **Semantic scene conditioning** — CLIP-based scene embeddings from SARLANG-1M enable adaptive despeckling (aggressive smoothing for water, edge preservation for urban areas)
4. **SAR-specific losses** — L1 + SSIM + Edge preservation loss, replacing VGG perceptual loss which is meaningless for SAR

### Key Contributions

- **First language-informed SAR despeckling framework** — scene semantics from SARLANG-1M's 1M+ expert-annotated captions guide the restoration process via Adaptive Instance Normalization (AdaIN)
- **Wavelet conditioning for ESRGAN** — novel combination not in existing literature, providing explicit frequency separation for joint despeckling + SR
- **Trained and evaluated on real NISAR L-band data** — NASA-ISRO's newest SAR satellite (launched 2025), not just synthetic benchmarks
- **SAR-specific loss design** — ENL-aware + edge-preserving losses replace optical-pretrained perceptual losses

## Architecture
```
Input (LR + Speckle) ──► DWT ──► 4 Subbands ──► Conv
                                                   │
                              CLIP Scene Embed ─────┤
                                                    ▼
                                    Conditioned RRDB × 16
                                    (AdaIN at each block)
                                                    │
                                                    ▼
                                    Pixel Shuffle Upsample
                                                    │
                                                    ▼
                                    Output (HR + Despeckled)
```

## Results

Evaluated on 500 test patches from NISAR L-band GCOV data:

| Metric | Bicubic | SAR-WaveSRNet | Improvement |
|--------|---------|---------------|-------------|
| PSNR ↑ | 11.83 dB | **14.80 dB** | +3.0 dB |
| SSIM ↑ | 0.318 | **0.359** | +13% |
| ENL ↑  | 5.77 | **24.88** | 4.3× |
| EPD →1 | 1.56 | **0.81** | Much closer to ideal |

## Dataset

### NISAR (Training Data)
- Source: [NASA-ISRO SAR Mission](https://nisar.jpl.nasa.gov/) via [ASF DAAC](https://asf.alaska.edu/)
- Product: L2 GCOV (Geocoded Covariance), L-band, HH polarization
- 19,286 patches of 128×128 pixels extracted with 50% overlap
- Synthetic Gamma speckle (L=1) added to create noisy/clean training pairs

### SARLANG-1M (Scene Conditioning)
- Source: [SARLANG-1M Benchmark](https://huggingface.co/datasets/YiminJimmy/SARLANG-1M) (IEEE TGRS 2026)
- 31,968 expert-annotated SAR scene captions analyzed to build semantic scene prototypes
- CLIP ViT-B/32 encodes both scene descriptions and SAR patches into shared embedding space
- 10 scene categories: urban_dense, urban_industrial, residential, water_body, port_harbor, vegetation, agricultural, desert_barren, coastal, mixed_terrain

## Installation
```bash
git clone https://github.com/yourusername/SAR-WaveSRNet.git
cd SAR-WaveSRNet
pip install torch torchvision pytorch_wavelets open_clip_torch h5py numpy PyYAML
```

## Usage

### Inference
```python
from models.generator import SARWaveSRNet
import torch

model = SARWaveSRNet(scale=2, embed_dim=512)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))
model.eval()

# input_patch: (1, 1, 64, 64) tensor
# scene_embed: (1, 512) CLIP embedding
output = model(input_patch, scene_embed)  # (1, 1, 128, 128)
```

### Training
See `notebooks/` for the complete training pipeline on Google Colab.

## Project Structure
```
SAR-WaveSRNet/
├── models/
│   ├── wavelet.py          # DWT/IDWT layers
│   ├── rrdb.py             # RRDB blocks
│   ├── generator.py        # Scene-conditioned generator (AdaIN + wavelet + RRDB)
│   └── discriminator.py    # PatchGAN discriminator
├── losses/
│   └── sar_loss.py         # L1 + SSIM + Edge + ENL losses
├── metrics/
│   └── evaluate.py         # PSNR, SSIM, ENL, EPD
├── data/
│   ├── dataset.py          # PyTorch Dataset with speckle injection
│   ├── preprocess.py       # NISAR HDF5 → patches
│   └── scene_encoder.py    # CLIP scene embedding pipeline
├── configs/
│   └── default.yaml        # Training configuration
├── checkpoints/
│   └── best_model.pth      # Trained weights
├── inference.py            # Inference script
└── README.md
```

## Citation

If you use this work, please cite:
```bibtex
@misc{sarwavesrnet2026,
  title={SAR-WaveSRNet: Joint SAR Despeckling and Super-Resolution via 
         Wavelet-Conditioned ESRGAN with Semantic Scene Conditioning},
  year={2026},
  note={Uses NISAR L-band data and SARLANG-1M scene embeddings}
}
```

### Acknowledgments

- [NISAR Mission](https://nisar.jpl.nasa.gov/) — NASA/ISRO for open SAR data
- [SARLANG-1M](https://arxiv.org/abs/2504.03254) — Wei et al. (IEEE TGRS 2026) for SAR vision-language benchmark
- [ESRGAN](https://github.com/xinntao/ESRGAN) — Wang et al. for the RRDB architecture
- [OpenCLIP](https://github.com/mlfoundations/open_clip) — CLIP implementation

## License

MIT License
