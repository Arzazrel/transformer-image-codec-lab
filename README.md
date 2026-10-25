# transformer-image-codec-lab

A hands-on laboratory for **learning and experimenting with Transformers** by
applying them to a concrete, non-trivial task: **image compression** via a
**Vision Transformer (ViT) autoencoder**.

The repository is meant both as a study path and as a portfolio piece: the code
is commented line by line, the architectures are fully parametrizable, and a
benchmark harness compares several models against each other and against JPEG.

> Methodological note. The goal is *educational and architectural*: to understand
> patch embedding, self-attention, and the encoder/bottleneck/decoder structure.
> A ViT autoencoder is **not** state of the art for image compression (that would
> be convolutional autoencoders with *entropy coding*, e.g. Balle 2017, Minnen
> 2018). This lab favors understanding the Transformer over codec optimality.

---

## Repository layout

| File | Role |
|------|------|
| `transformer_blocks.py` | Building blocks (patch embedding, positional encoding, multi-head attention, MLP, transformer block), explained and with **incremental tests** that show how tensor *shapes* evolve. |
| `model.py` | The **complete, parametrizable** ViT autoencoder: encoder -> z-layer (bottleneck) -> decoder, with *optional* code quantization. |
| `benchmark.py` | Defines several models, trains them, measures performance, and compares them with each other and with **JPEG** (rate-distortion curve). |
| `codec.py` | Practical use of a trained model to **compress/decompress** a single image. |

---

## Architecture

```
ENCODER (compressor)
  image (B,C,H,W)
    -> PatchEmbedding + PositionalEncoding
    -> depth_enc x TransformerBlock
    -> linear projection -> Z-LAYER   <- compressed code

[optional] quantization of Z to z_bits bits

DECODER (decompressor)
  Z-LAYER
    -> projection + PositionalEncoding
    -> depth_dec x TransformerBlock
    -> linear head -> patch reassembly -> image (B,C,H,W)
```

The **compression ratio** is governed by `patch_size` (number of tokens) and by
`z_dim` (bottleneck width per token). Number of code values =
`num_patches x z_dim`.

### Main parameters (`ViTAutoencoderConfig`)

| Field | Meaning |
|-------|---------|
| `img_size`, `in_channels`, `patch_size` | input and patch geometry |
| `embed_dim`, `num_heads`, `mlp_ratio` | transformer capacity |
| `z_dim` | **compression dimension** (bottleneck) |
| `depth_enc`, `depth_dec` | encoder and decoder depth |
| `quantize`, `z_bits` | optional code quantization (see below) |

### Dimensionality reduction vs. real compression

With `quantize=False` (default) the code `z` stays in `float32`: this is
**dimensionality reduction**, not bit compression. For a **real bit-rate (bpp)**
and a fair comparison with JPEG, set `quantize=True` with a bit budget `z_bits`;
quantization uses a *Straight-Through Estimator* to remain differentiable during
training. Quantization is included as an option to explore further.

---

## Metrics

- **Timing** of compression (encode) and decompression (decode), per image, with
  CUDA synchronization and warmup.
- **Compression ratio**: dimensionality reduction always; `bpp` and bit ratio when
  `quantize=True`.
- **Reconstruction fidelity**: MSE, **PSNR** (dB), **SSIM**. *(Accuracy is not
  used: it is a classification metric, unsuitable for reconstruction.)*
- **Complexity**: parameter count and training time.
- **JPEG baseline**: PSNR as a function of bpp (rate-distortion curve).

---

## Datasets

- **CIFAR-10** (32x32) - auto-downloaded, for fast iteration.
- **Kodak** (24 uncompressed PNGs) - the de facto standard for R-D curves on
  natural images; downloaded on demand and cropped to `img_size`.

PNG/lossless so that model distortion is not confounded with the distortion of a
pre-existing compression.

---

## Requirements

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision pillow numpy
```

Designed to run on an **NVIDIA RTX GPU via WSL2**; it also works on CPU (slower).
Check the GPU with `python -c "import torch; print(torch.cuda.is_available())"`.

---

## Quick start

```bash
# 1) understand the blocks (shapes, step by step)
python transformer_blocks.py

# 2) build the model and inspect parameters/ratios
python model.py

# 3) train and compare the models + JPEG
python benchmark.py --dataset cifar --epochs 5
python benchmark.py --dataset kodak --epochs 50 --img_size 256

# 4) use a trained model on your own image
python codec.py roundtrip --ckpt ckpt_base-z16.pt --in photo.png --out reconstructed.png
```

`benchmark.py` saves a `ckpt_<name>.pt` checkpoint for each configuration,
reusable by `codec.py`.

---

## Possible extensions

- Quantization + an entropy model over the code for a genuinely competitive codec.
- Windowed SSIM (`pytorch-msssim`) and MS-SSIM.
- A perceptual loss (LPIPS) in addition to MSE.
- Variable-resolution autoencoder (patches over full-resolution images).
- Comparison with WebP/AVIF in addition to JPEG.

---

## Author
*Laboratory by Alessandro Diana with Claude
