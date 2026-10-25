"""
transformer_blocks.py

Blocchi costitutivi di un Vision Transformer (ViT) usato come AUTOENCODER per la compressione di immagini.
Se il file viene eseguito direttamente (python transformer_blocks.py), una serie di test con `print` mostra passo-passo come i tensori cambiano forma mentre attraversano i blocchi, fino a comporre l'encoder completo.
La pipeline logica che costruiamo e':
    immagine  ->  patch  ->  embedding  ->  [+ positional] ->  N x TransformerBlock  ->  rappresentazione latente
Lo stesso meccanismo, "a specchio", ricostruira' l'immagine nel decoder (implementato in model.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
-- PATCH EMBEDDING --
Trasforma un'immagine (B, C, H, W) in una sequenza di "token", uno per patch quadrata di lato `patch_size`.
Conv2d con kernel=stride=patch_size
E' il trucco standard del ViT -> una convoluzione con kernel pari alla patch e stride uguale al kernel equivale: 
- ritagliare le patch non sovrapposte
- proiettarle linearmente in uno spazio di dimensione `embed_dim`. 
Una sola operazione, efficiente su GPU.
"""
class PatchEmbedding(nn.Module):
    # default size for 
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        super().__init__()
        assert img_size % patch_size == 0, \
            "img_size deve essere divisibile per patch_size"

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size # numero di patch lungo un lato
        self.num_patches = self.grid_size ** 2  # numero di patch in totale

        # proiezione lineare di ogni patch -> vettore di dimensione embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim,kernel_size=patch_size, stride=patch_size,)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                 # (B, embed_dim, grid, grid)
        x = x.flatten(2)                 # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)            # (B, num_patches, embed_dim)  <- sequenza di token
        return x

"""
-- POSITIONAL ENCODING --
informazione posizionale apprendibile dal modello (un parametro per posizione),scelta comune nei ViT e piu' semplice del positional sinusoidale.
"""
class PositionalEncoding(nn.Module):
    
    def __init__(self, num_tokens, embed_dim):
        super().__init__()
        # un vettore imparabile per ogni posizione della sequenza
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, num_tokens, embed_dim)
        return x + self.pos_embed

"""
-- MULTI-HEAD SELF-ATTENTION --
Ogni token "guarda" tutti gli altri e aggrega informazione pesata dalla similarita' (attention).
Multi-head: proiettiamo in `num_heads` sottospazi indipendenti, calcoliamo l'attention in ciascuno e concateniamo. 
Teste diverse possono specializzarsi su relazioni diverse (es. bordi, texture, colore).
"""
class MultiHeadSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.0, proj_dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, \
            "embed_dim deve essere divisibile per num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5          # 1/sqrt(d_k): stabilizza i gradienti

        # una sola proiezione lineare produce Q, K, V insieme (piu' efficiente)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, D = x.shape                           # batch, token, dim

        # (B, N, 3D) -> (3, B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]            # ciascuno (B, heads, N, head_dim)

        # punteggi di attenzione: quanto ogni token e' rilevante per ogni altro
        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
        attn = attn.softmax(dim=-1)                      # normalizzazione su ogni riga
        attn = self.attn_drop(attn)

        # media pesata dei valori
        out = attn @ v                                   # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)       # riconcatena le teste
        out = self.proj_drop(self.proj(out))
        return out

"""
-- MLP (FEED-FORWARD) del blocco transformer --
Rete feed-forward applicata a ogni token indipendentemente.
Espande la dimensione (mlp_ratio) e la riporta a embed_dim: aggiunge capacita' non lineare dopo la miscelazione operata dall'attention.
"""
class MLP(nn.Module):
    
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)         # hidden size
        self.fc1 = nn.Linear(embed_dim, hidden)     # input layer
        self.act = nn.GELU()                        # GELU: standard nei transformer
        self.fc2 = nn.Linear(hidden, embed_dim)     # output layer
        self.drop = nn.Dropout(dropout)             # dropout

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

"""
-- BLOCCO TRANSFORMER --
Blocco canonico: due sotto-strati (attention e MLP), ciascuno con connessione residua e LayerNorm applicato PRIMA (pre-norm, piu' stabile in training rispetto al post-norm dell'articolo originale).
    x = x + Attn(Norm(x))
    x = x + MLP(Norm(x))
"""
class TransformerBlock(nn.Module):
    
    def __init__(self, embed_dim, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)            # layer normalization
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads,attn_dropout=dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)            # layer normalization
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)   # MLP

    def forward(self, x):
        x = x + self.attn(self.norm1(x))            # residuo 1
        x = x + self.mlp(self.norm2(x))             # residuo 2
        return x


# ---------------------------------------------------------------------------
# TEST INCREMENTALI
# ---------------------------------------------------------------------------
# Eseguendo `python transformer_blocks.py` vediamo, blocco dopo blocco, come un batch di immagini CIFAR-like (3x32x32) viene trasformato fino a diventare
# una sequenza latente pronta per il bottleneck del compressore.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}\n")

    # batch finto: 2 immagini RGB 32x32
    B, C, H, W = 2, 3, 32, 32
    img = torch.randn(B, C, H, W, device=device)
    print(f"[0] immagine in input           : {tuple(img.shape)}")

    # --- patch embedding ---
    patch = PatchEmbedding(img_size=H, patch_size=4, in_channels=C, embed_dim=128).to(device)
    tokens = patch(img)
    print(f"[1] dopo PatchEmbedding         : {tuple(tokens.shape)} "
          f"(Patch size = 4, num_patches={patch.num_patches}, embed_dim=128)")

    # --- positional encoding ---
    pos = PositionalEncoding(num_tokens=patch.num_patches, embed_dim=128).to(device)
    tokens = pos(tokens)
    print(f"[2] dopo PositionalEncoding     : {tuple(tokens.shape)} (forma invariata)")

    # --- una singola testa/attention ---
    attn = MultiHeadSelfAttention(embed_dim=128, num_heads=4).to(device)
    out_attn = attn(tokens)
    print(f"[3] dopo Multi-Head Attention   : {tuple(out_attn.shape)}")

    # --- MLP ---
    mlp = MLP(embed_dim=128).to(device)
    out_mlp = mlp(out_attn)
    print(f"[4] dopo MLP                    : {tuple(out_mlp.shape)}")

    # --- blocco completo ---
    block = TransformerBlock(embed_dim=128, num_heads=4).to(device)
    out_block = block(tokens)
    print(f"[5] dopo 1 TransformerBlock     : {tuple(out_block.shape)}")

    # --- stack di blocchi = encoder ---
    depth = 4
    encoder = nn.Sequential(*[TransformerBlock(128, 4) for _ in range(depth)]).to(device)
    latent = encoder(tokens)
    print(f"[6] dopo {depth} blocchi (encoder) : {tuple(latent.shape)}")

    # conteggio parametri dell'encoder giocattolo
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n[info] parametri encoder: {n_params:,}")
    print("[ok] i blocchi compongono correttamente un encoder transformer.")
