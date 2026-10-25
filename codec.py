"""
codec.py

Uso PRATICO di un modello gia' allenato (checkpoint prodotto da benchmark.py) per comprimere e decomprimere una singola immagine.
Flusso:
1) carica il checkpoint (pesi + configurazione)
2) compress()   : immagine PNG -> codice z salvato su file .npz
3) decompress() : codice z .npz -> immagine PNG ricostruita
4) report di PSNR/SSIM e dimensioni a confronto

ESEMPI DA RIGA DI COMANDO
# comprimere
python codec.py compress   --ckpt ckpt_base-z16.pt --in foto.png --code foto.npz
# decomprimere
python codec.py decompress --ckpt ckpt_base-z16.pt --code foto.npz --out ricostruita.png
# ciclo completo con metriche
python codec.py roundtrip  --ckpt ckpt_base-z16.pt --in foto.png --out ricostruita.png

NOTA: l'immagine viene ridimensionata/ritagliata a img_size del modello.
"""

import argparse
import os

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from model import ViTAutoencoder, ViTAutoencoderConfig


# ---------------------------------------------------------------------------

# Ricostruisce il modello dalla configurazione salvata e carica i pesi.
def load_model(ckpt_path, device):
    
    ckpt = torch.load(ckpt_path, map_location=device)   # load weight
    cfg = ViTAutoencoderConfig(**ckpt["cfg"])
    model = ViTAutoencoder(cfg).to(device).eval()       # create new model
    model.load_state_dict(ckpt["state_dict"])           # put the weight in the model
    return model, cfg

# Carica un'immagine, la porta a RGB e al lato img_size, tensore in [0,1].
def load_image(path, img_size):
    
    img = Image.open(path).convert("RGB")
    tf = T.Compose([T.Resize(img_size), T.CenterCrop(img_size), T.ToTensor()])
    return tf(img).unsqueeze(0)             # (1, C, H, W)

# Salva un tensore (1,C,H,W) o (C,H,W) in [0,1] come PNG.
def save_image(tensor, path):

    if tensor.dim() == 4:
        tensor = tensor[0]
    T.ToPILImage()(tensor.clamp(0, 1).cpu()).save(path)


# ---------------------------------------------------------------------------
# 
@torch.no_grad()
def compress(model, cfg, img_path, code_path, device):
    x = load_image(img_path, cfg.img_size).to(device)
    z = model.encode(x)                     # (1, N, z_dim)
    arr = z.cpu().numpy().astype(np.float32)
    # se il modello e' quantizzato possiamo salvare in interi su z_bits bit
    if cfg.quantize:
        levels = 2 ** cfg.z_bits - 1
        arr = np.round(arr * levels).astype(np.uint16)
    np.savez_compressed(code_path, z=arr, quantized=cfg.quantize, z_bits=cfg.z_bits)
    print(f"[compress] codice salvato in {code_path}  shape={arr.shape}")
    return z


@torch.no_grad()
def decompress(model, cfg, code_path, out_path, device):
    data = np.load(code_path)
    arr = data["z"].astype(np.float32)
    if bool(data["quantized"]):
        levels = 2 ** int(data["z_bits"]) - 1
        arr = arr / levels                  # riporta i livelli interi in [0,1]
    z = torch.from_numpy(arr).to(device)
    x_hat = model.decode(z)
    save_image(x_hat, out_path)
    print(f"[decompress] immagine ricostruita salvata in {out_path}")
    return x_hat


@torch.no_grad()
def roundtrip(model, cfg, img_path, out_path, device):
    """Comprime e decomprime, poi riporta fedelta' e rapporto di dimensioni."""
    from benchmark import psnr, ssim

    x = load_image(img_path, cfg.img_size).to(device)
    z = model.encode(x)
    x_hat = model.decode(z)
    save_image(x_hat, out_path)

    # dimensioni a confronto (float32 per il codice, salvo quantizzazione)
    img_vals = cfg.in_channels * cfg.img_size ** 2
    code_vals = model.code_size_floats()
    bytes_img = img_vals * 4
    bytes_code = code_vals * (np.ceil(cfg.z_bits / 8) if cfg.quantize else 4)

    print(f"[roundtrip] PSNR = {psnr(x, x_hat):.2f} dB   SSIM = {ssim(x, x_hat):.3f}")
    print(f"[roundtrip] valori immagine={img_vals}  valori codice={code_vals}  "
          f"riduzione={img_vals/code_vals:.2f}x")
    print(f"[roundtrip] byte (grezzi) immagine={bytes_img}  codice={int(bytes_code)}  "
          f"rapporto={bytes_img/bytes_code:.2f}x")
    if not cfg.quantize:
        print("[nota] modello non quantizzato: il 'rapporto' e' dimensionale, "
              "non una compressione in bit reale.")
    return x_hat


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["compress", "decompress", "roundtrip"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out")
    ap.add_argument("--code")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.ckpt, device)

    if args.mode == "compress":
        compress(model, cfg, args.inp, args.code, device)
    elif args.mode == "decompress":
        decompress(model, cfg, args.code, args.out, device)
    else:
        roundtrip(model, cfg, args.inp, args.out, device)
