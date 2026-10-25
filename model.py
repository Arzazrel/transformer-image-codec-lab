"""
model.py

Autoencoder basato su Vision Transformer (ViT) per la COMPRESSIONE di immagini. La rete e' completamente parametrizzabile per permettere di confrontare configurazioni diverse.

ARCHITETTURA
- ENCODER  (compressore)
    immagine (B,C,H,W)
    -> PatchEmbedding + PositionalEncoding
    -> depth_enc x TransformerBlock
    -> proiezione lineare su z_dim per token
    -> Z-LAYER  (il "codice" compresso)            <-- output compressore
  [opzionale] QUANTIZZAZIONE di Z  (quantize=True)

- DECODER  (decompressore)
    Z-LAYER
    -> proiezione lineare z_dim -> embed_dim + PositionalEncoding
    -> depth_dec x TransformerBlock
    -> testa lineare che rigenera i pixel di ogni patch
    -> ricomposizione dell'immagine (B,C,H,W)       <-- output decompressore

DIMENSIONE DELLA COMPRESSIONE
Il grado di compressione e' governato principalmente da:
    - patch_size   : patch piu' grandi -> meno token
    - z_dim        : dimensione del codice per token (il collo di bottiglia)
Il numero di valori nel codice e':  num_patches * z_dim.
Rapporto di riduzione dimensionale (float): (C*H*W) / (num_patches * z_dim)

Nota:
Finche' z resta in float32, questa e' RIDUZIONE DI DIMENSIONALITA', non compressione in bit: ogni valore occupa comunque 32 bit. 
Per un tasso di compressione REALE (bpp) e un confronto equo con JPEG occorre quantizzare z a pochi bit: lo si abilita con `quantize=True, z_bits=N`.
Di default quantize=False: il focus didattico e' sul transformer.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from transformer_blocks import (
    PatchEmbedding,
    PositionalEncoding,
    TransformerBlock,
)


"""
Configurazione del modello: un unico oggetto parametrizza tutta la rete.
Cambiando i campi si generano architetture diverse da confrontare.
"""
@dataclass
class ViTAutoencoderConfig:
    img_size: int = 32          # lato immagine (assunta quadrata)
    in_channels: int = 3        # 3 = RGB
    patch_size: int = 4         # lato patch; influenza num_patches e compressione
    embed_dim: int = 128        # dimensione dei token nel transformer
    z_dim: int = 16             # dimensione del codice per token (BOTTLENECK)
    depth_enc: int = 4          # numero di blocchi transformer nell'encoder
    depth_dec: int = 4          # numero di blocchi transformer nel decoder
    num_heads: int = 4          # teste di attention
    mlp_ratio: float = 4.0      # fattore di espansione dell'MLP
    dropout: float = 0.0        # dropout (utile con dataset piccoli)
    # --- quantizzazione opzionale del codice z ---
    quantize: bool = False      # se True, z viene quantizzato a z_bits bit
    z_bits: int = 8             # bit per valore quando quantize=True


"""
Quantizzatore uniforme con Straight-Through Estimator (STE).
Quantizza z (atteso in [0,1] grazie alla Sigmoid del bottleneck) su 2**bits livelli uniformi.
Problema: la funzione di arrotondamento ha gradiente nullo quasi ovunque e bloccherebbe il training. 
Soluzione classica (STE): in forward si quantizza davvero, in backward si lascia passare il gradiente come se l'operazione fosse l'identita'. 
   `x + (q - x).detach()` ottiene esattamente questo.
"""
class UniformQuantizer(nn.Module):
    
    def __init__(self, bits=8):
        super().__init__()
        self.levels = 2 ** bits - 1

    def forward(self, x):
        if not self.training:
            # in inference quantizziamo in modo netto (nessun gradiente serve)
            return torch.round(x * self.levels) / self.levels
        q = torch.round(x * self.levels) / self.levels
        return x + (q - x).detach()                 # STE


# Modello completo
class ViTAutoencoder(nn.Module):
    def __init__(self, cfg: ViTAutoencoderConfig):
        super().__init__()
        self.cfg = cfg

        # ---------------- ENCODER ----------------
        self.patch_embed = PatchEmbedding(img_size=cfg.img_size, patch_size=cfg.patch_size,in_channels=cfg.in_channels, embed_dim=cfg.embed_dim,)
        self.num_patches = self.patch_embed.num_patches
        self.patch_dim = cfg.in_channels * cfg.patch_size ** 2   # pixel per patch

        self.enc_pos = PositionalEncoding(self.num_patches, cfg.embed_dim)
        self.enc_blocks = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.depth_enc)
        ])
        self.enc_norm = nn.LayerNorm(cfg.embed_dim)

        # proiezione al bottleneck. Sigmoid -> z in [0,1]: utile sia per la quantizzazione sia per stabilizzare la scala del codice.
        self.to_z = nn.Sequential(nn.Linear(cfg.embed_dim, cfg.z_dim),nn.Sigmoid(),)

        # ------------- QUANTIZZATORE -------------
        self.quantizer = UniformQuantizer(cfg.z_bits) if cfg.quantize else None

        # ---------------- DECODER ----------------
        self.from_z = nn.Linear(cfg.z_dim, cfg.embed_dim)
        self.dec_pos = PositionalEncoding(self.num_patches, cfg.embed_dim)
        self.dec_blocks = nn.ModuleList([
            TransformerBlock(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.depth_dec)
        ])
        self.dec_norm = nn.LayerNorm(cfg.embed_dim)

        # testa che da ogni token ricostruisce i pixel della sua patch
        self.to_pixels = nn.Linear(cfg.embed_dim, self.patch_dim)

    # ---- encoder: immagine -> codice z ----
    def encode(self, x):
        x = self.patch_embed(x)                     # (B, N, embed_dim)
        x = self.enc_pos(x)
        for blk in self.enc_blocks:
            x = blk(x)
        x = self.enc_norm(x)
        z = self.to_z(x)                            # (B, N, z_dim)  in [0,1]
        if self.quantizer is not None:
            z = self.quantizer(z)
        return z

    # ---- decoder: codice z -> immagine ----
    def decode(self, z):
        x = self.from_z(z)                          # (B, N, embed_dim)
        x = self.dec_pos(x)
        for blk in self.dec_blocks:
            x = blk(x)
        x = self.dec_norm(x)
        patches = self.to_pixels(x)                 # (B, N, patch_dim)
        return self._unpatchify(patches)

    # ---- ricomposizione patch -> immagine ----
    def _unpatchify(self, patches):
        cfg = self.cfg
        g = cfg.img_size // cfg.patch_size          # patch per lato
        p = cfg.patch_size
        B = patches.shape[0]
        # (B, N, C*p*p) -> immagine (B, C, H, W)
        x = patches.reshape(B, g, g, cfg.in_channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, cfg.in_channels, g * p, g * p)
        return torch.sigmoid(x)                     # pixel in [0,1]

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    # ---- utilita' per il reporting ----
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def code_size_floats(self):
        """Numero di valori scalari nel codice z (per il calcolo del rapporto)."""
        return self.num_patches * self.cfg.z_dim

    def dimensional_reduction_ratio(self):
        """(valori immagine) / (valori codice). >1 = riduzione, non ancora bit."""
        cfg = self.cfg
        img_vals = cfg.in_channels * cfg.img_size ** 2
        return img_vals / self.code_size_floats()


# ---------------------------------------------------------------------------
# Test rapido: costruzione, forward, conteggio parametri e rapporto.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}\n")

    cfg = ViTAutoencoderConfig(img_size=32, patch_size=4, embed_dim=128,
                               z_dim=16, depth_enc=4, depth_dec=4)
    model = ViTAutoencoder(cfg).to(device)

    x = torch.rand(2, 3, 32, 32, device=device)     # batch finto in [0,1]
    x_hat, z = model(x)

    print(f"input            : {tuple(x.shape)}")
    print(f"codice z         : {tuple(z.shape)}  (num_patches={model.num_patches}, z_dim={cfg.z_dim})")
    print(f"ricostruzione    : {tuple(x_hat.shape)}")
    print(f"parametri        : {model.num_parameters():,}")
    print(f"valori immagine  : {3*32*32}")
    print(f"valori codice z  : {model.code_size_floats()}")
    print(f"riduzione dim.   : {model.dimensional_reduction_ratio():.2f}x "
          f"(float32; per bpp reale serve quantize=True)")

    # prova con quantizzazione attiva
    cfg_q = ViTAutoencoderConfig(quantize=True, z_bits=6)
    model_q = ViTAutoencoder(cfg_q).to(device)
    _, zq = model_q.eval()(x)
    print(f"\n[quant] z_bits=6 -> livelli distinti in z: "
          f"{zq.unique().numel()} (quantizzazione attiva)")
