"""
benchmark.py

Definisce piu' configurazioni di ViT-autoencoder, le allena su un dataset, ne misura le prestazioni e le confronta tra loro e con JPEG.

METRICHE MISURATE
- tempo di compressione (encode) e decompressione (decode)
- rapporto di compressione:
    * riduzione dimensionale (float32)  -> sempre
    * bpp reale e rapporto in bit       -> solo se quantize=True
- fedelta' di ricostruzione: MSE, PSNR (dB), SSIM
- complessita': numero di parametri e tempo di training

Per la fedelta' di un'immagine ricostruita si usano PSNR (errore in dB, piu' alto = meglio) e SSIM (similarita' strutturale percettiva in [0,1], piu' alto = meglio).

CONFRONTO CON JPEG
JPEG viene valutato su un intervallo di "quality" per tracciare la curva rate-distortion (PSNR in funzione dei bit-per-pixel). 
Il confronto con la rete e' sensato SOLO in bpp, quindi e' pienamente equo quando la rete usa la quantizzazione (quantize=True).

VISUALIZZAZIONE FINALE
Al termine di training e confronto, viene aperta (o salvata, se manca un display)
una griglia che mostra, per alcune immagini casuali del test set, l'originale e
le ricostruzioni di tutti i modelli e di JPEG, annotate con i dati del modello e
i tempi di codifica/decodifica.

DATASET
- CIFAR-10 : 32x32, scaricato in automatico, per iterare in fretta.
- Kodak    : 24 PNG non compressi, standard per le curve R-D su immagini naturali. Scaricati on-demand e ritagliati a 'img_size'.
"""

import io
import os
import time
import random
import urllib.request

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from PIL import Image

from model import ViTAutoencoder, ViTAutoencoderConfig

# ----------------------------------- METRICHE -----------------------------------

# Peak Signal-to-Noise Ratio in dB. Immagini attese in [0, max_val].
def psnr(x, x_hat, max_val=1.0):
    mse = torch.mean((x - x_hat) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(max_val ** 2 / mse)).item()

"""
SSIM globale semplificato (medie/varianze sull'intera immagine anziche' con finestra scorrevole). 
Sufficiente per confronti relativi; per un valore conforme allo standard usare pytorch-msssim.
Input: (B,C,H,W) in [0,1]. Ritorna la media sul batch.
"""
def ssim(x, x_hat, C1=0.01 ** 2, C2=0.03 ** 2):
    
    mu_x = x.mean(dim=(1, 2, 3))
    mu_y = x_hat.mean(dim=(1, 2, 3))
    var_x = x.var(dim=(1, 2, 3), unbiased=False)
    var_y = x_hat.var(dim=(1, 2, 3), unbiased=False)
    cov = ((x - mu_x[:, None, None, None]) *
           (x_hat - mu_y[:, None, None, None])).mean(dim=(1, 2, 3))
    s = (((2 * mu_x * mu_y + C1) * (2 * cov + C2)) /
         ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2)))
    return s.mean().item()

# ----------------------------------- DATASET -----------------------------------

# CIFAR-10 in [0,1]. Nessuna normalizzazione: vogliamo pixel reali.
def get_cifar10_loaders(batch_size=128, img_size=32, root="./data"):
    tf = T.Compose([T.Resize(img_size), T.ToTensor()])
    train = torchvision.datasets.CIFAR10(root, train=True, download=True, transform=tf)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True, transform=tf)
    return (DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=2),
            DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=2))

"""
Kodak: 24 immagini PNG non compresse (kodim01..kodim24).
Le scarichiamo una volta in 'root' e, per semplificare l'autoencoder a risoluzione fissa, ne ritagliamo un quadrato centrale di lato img_size.
"""
class KodakDataset(Dataset):
    
    BASE = "https://r0k.us/graphics/kodak/kodak/"

    def __init__(self, img_size=256, root="./data/kodak"):
        os.makedirs(root, exist_ok=True)
        self.paths = []
        for i in range(1, 25):
            fn = f"kodim{i:02d}.png"
            fp = os.path.join(root, fn)
            if not os.path.exists(fp):
                try:
                    urllib.request.urlretrieve(self.BASE + fn, fp)
                except Exception as e:
                    print(f"[warn] download {fn} fallito: {e}")
                    continue
            self.paths.append(fp)
        self.tf = T.Compose([T.CenterCrop(img_size), T.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), 0          # label fittizia per compatibilita'


# ----------------------------------- TRAINING -----------------------------------

# Allena l'autoencoder con loss MSE. Ritorna il tempo di training (s).
def train_model(model, loader, device, epochs=10, lr=1e-3, log_every=100):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    t0 = time.time()
    for ep in range(epochs):
        running = 0.0
        for step, (x, _) in enumerate(loader):
            x = x.to(device)
            x_hat, _ = model(x)
            loss = loss_fn(x_hat, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            if log_every and step % log_every == 0:
                print(f"  ep {ep+1}/{epochs}  step {step}  loss {loss.item():.5f}")
        print(f"[ep {ep+1}] loss media = {running/len(loader):.5f}")
    return time.time() - t0


# ----------------------------------- VALUTAZIONE DI UN MODELLO -----------------------------------

"""
Calcola PSNR/SSIM medi e i tempi medi di encode/decode per immagine.
I tempi sono misurati con sincronizzazione CUDA ed escludendo un warmup iniziale (le prime iterazioni includono overhead di inizializzazione).
"""
@torch.no_grad()
def evaluate_model(model, loader, device, warmup=2):
    
    model.to(device).eval()
    tot_psnr = tot_ssim = n = 0
    enc_times, dec_times = [], []

    for i, (x, _) in enumerate(loader):
        x = x.to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        z = model.encode(x)
        if device == "cuda":
            torch.cuda.synchronize()
        enc_times.append((time.time() - t) / x.size(0))

        t = time.time()
        x_hat = model.decode(z)
        if device == "cuda":
            torch.cuda.synchronize()
        dec_times.append((time.time() - t) / x.size(0))

        tot_psnr += psnr(x, x_hat) * x.size(0)
        tot_ssim += ssim(x, x_hat) * x.size(0)
        n += x.size(0)

    enc = sum(enc_times[warmup:]) / max(1, len(enc_times[warmup:]))
    dec = sum(dec_times[warmup:]) / max(1, len(dec_times[warmup:]))

    cfg = model.cfg
    out = {
        "params": model.num_parameters(),
        "psnr": tot_psnr / n,
        "ssim": tot_ssim / n,
        "enc_s_per_img": enc,
        "dec_s_per_img": dec,
        "dim_reduction": model.dimensional_reduction_ratio(),
    }
    # bpp reale solo se il codice e' quantizzato
    if cfg.quantize:
        code_bits = model.code_size_floats() * cfg.z_bits
        out["bpp"] = code_bits / (cfg.img_size ** 2)
        out["bit_ratio"] = (cfg.in_channels * cfg.img_size ** 2 * 8) / code_bits
    return out


"""
Per ogni livello di qualita' JPEG, ricodifica le immagini in memoria e misura bpp medio e PSNR medio. Restituisce punti (bpp, psnr) = curva R-D.
"""
@torch.no_grad()
def jpeg_rate_distortion(loader, device, qualities=(10, 25, 50, 75, 90)):
    
    points = []
    for q in qualities:
        tot_psnr = tot_bpp = n = 0
        for x, _ in loader:
            for img in x:                                   # (C,H,W) in [0,1]
                pil = T.ToPILImage()(img.cpu())
                buf = io.BytesIO()
                pil.save(buf, format="JPEG", quality=q)
                bits = buf.tell() * 8
                bpp = bits / (img.shape[1] * img.shape[2])
                rec = T.ToTensor()(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
                tot_psnr += psnr(img.unsqueeze(0), rec.unsqueeze(0))
                tot_bpp += bpp
                n += 1
        points.append({"quality": q, "bpp": tot_bpp / n, "psnr": tot_psnr / n})
        print(f"[jpeg q={q:3d}]  bpp={tot_bpp/n:.3f}  psnr={tot_psnr/n:.2f} dB")
    return points


# ----------------------------------- VISUALIZZAZIONE QUALITATIVA -----------------------------------

"""
Comprime/decomprime in JPEG una singola immagine (C,H,W in [0,1]).
Ritorna (ricostruzione come tensore, bpp effettivo). Serve a mostrare JPEG
nella griglia a un bitrate confrontabile con quello dei modelli.
"""
def _jpeg_reconstruct(img_chw, quality):
    pil = T.ToPILImage()(img_chw.cpu().clamp(0, 1))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    bpp = (buf.tell() * 8) / (img_chw.shape[1] * img_chw.shape[2])
    rec = T.ToTensor()(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return rec, bpp


"""
Sceglie la quality JPEG il cui bpp medio e' piu' vicino a target_bpp.
Confrontare JPEG e rete a bitrate simile rende il paragone visivo equo anziche'
arbitrario. La stima usa un solo batch del loader (sufficiente e veloce).
"""
def _nearest_jpeg_quality(target_bpp, loader, qualities=(10, 20, 30, 40, 50, 60, 70, 80, 90)):
    best_q, best_gap = qualities[len(qualities) // 2], float("inf")
    for q in qualities:
        tot_bpp = n = 0
        for x, _ in loader:
            for img in x:
                _, bpp = _jpeg_reconstruct(img, q)
                tot_bpp += bpp
                n += 1
            break                       # un solo batch basta per la stima
        gap = abs((tot_bpp / max(1, n)) - target_bpp)
        if gap < best_gap:
            best_q, best_gap = q, gap
    return best_q


"""
Mostra una griglia: originale vs ricostruzioni di tutti i modelli vs JPEG, su
immagini casuali del test set.

- righe   = immagini campione
- colonne = [Originale | modello_1 | ... | modello_k | JPEG]
Ogni cella e' annotata con PSNR e, per i modelli, i tempi enc/dec dell'immagine
mostrata (ricalcolati sul singolo campione, cosi' i numeri si riferiscono
esattamente a quell'immagine). I titoli di colonna (prima riga) riportano i
parametri del modello.

Se non c'e' un display (tipico su WSL senza X server) la figura viene salvata in
save_path invece di essere aperta a schermo.
"""
@torch.no_grad()
def visualize_reconstructions(models, loader, device, n_samples=4,
                              jpeg_quality=None, target_bpp=None,
                              save_path="reconstructions.png"):
    import matplotlib
    # backend non interattivo se non e' disponibile una finestra grafica
    interactive = matplotlib.get_backend().lower() not in ("agg", "pdf", "svg", "ps")
    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- raccogli n_samples immagini casuali dal test set ---
    all_imgs = torch.cat([x for x, _ in loader], dim=0)
    k = min(n_samples, all_imgs.size(0))
    idx = random.sample(range(all_imgs.size(0)), k=k)
    samples = all_imgs[idx].to(device)                  # (S, C, H, W)

    # --- scelta della quality JPEG ---
    if jpeg_quality is None:
        jpeg_quality = _nearest_jpeg_quality(target_bpp, loader) if target_bpp else 50

    model_names = list(models.keys())
    n_cols = 1 + len(model_names) + 1                   # originale + modelli + jpeg
    n_rows = samples.size(0)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.9 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)                      # normalizza a 2D

    def _show(ax, img_chw, title):
        ax.imshow(img_chw.cpu().clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    for r in range(n_rows):
        orig = samples[r]                               # (C,H,W)

        # colonna 0: originale
        _show(axes[r, 0], orig, "Original" if r == 0 else "")

        # colonne dei modelli
        for c, name in enumerate(model_names, start=1):
            model = models[name].to(device).eval()
            x_in = orig.unsqueeze(0)                    # (1,C,H,W)

            if device == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            z = model.encode(x_in)
            if device == "cuda":
                torch.cuda.synchronize()
            enc_ms = (time.time() - t) * 1e3

            t = time.time()
            x_hat = model.decode(z)
            if device == "cuda":
                torch.cuda.synchronize()
            dec_ms = (time.time() - t) * 1e3

            p = psnr(x_in, x_hat)
            cfg = model.cfg
            head = (f"{name}\nz={cfg.z_dim} d={cfg.depth_enc}/{cfg.depth_dec} "
                    f"emb={cfg.embed_dim}\n") if r == 0 else ""
            _show(axes[r, c], x_hat[0],
                  f"{head}PSNR {p:.1f} dB\nenc {enc_ms:.1f} / dec {dec_ms:.1f} ms")

        # ultima colonna: JPEG
        rec, bpp = _jpeg_reconstruct(orig, jpeg_quality)
        p = psnr(orig.unsqueeze(0), rec.unsqueeze(0).to(device))
        head = f"JPEG (q={jpeg_quality})\nbpp {bpp:.2f}\n" if r == 0 else ""
        _show(axes[r, n_cols - 1], rec.to(device), f"{head}PSNR {p:.1f} dB")

    fig.suptitle("Original vs reconstructions (random test images)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if interactive:
        plt.show()
    else:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"[viz] nessun display disponibile: figura salvata in {save_path}")
    plt.close(fig)


# ORCHESTRAZIONE: definisci modelli -> allena -> confronta
def run(dataset="cifar", epochs=5, batch_size=128, img_size=None, device=None,
        n_viz=4):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}\n")

    # ---- dataset ----
    if dataset == "cifar":
        img_size = img_size or 32
        train_loader, test_loader = get_cifar10_loaders(batch_size, img_size)
    elif dataset == "kodak":
        img_size = img_size or 256
        ds = KodakDataset(img_size=img_size)
        train_loader = DataLoader(ds, batch_size=4, shuffle=True)
        test_loader = DataLoader(ds, batch_size=4, shuffle=False)
    else:
        raise ValueError("dataset deve essere 'cifar' o 'kodak'")

    # ---- MODELLI DA CONFRONTARE ----
    # Modificare liberamente per esplorare lo spazio delle architetture.
    configs = {
        "tiny-z8":   ViTAutoencoderConfig(img_size=img_size, patch_size=4,embed_dim=96, z_dim=8,depth_enc=3, depth_dec=3, num_heads=4),
        "base-z16":  ViTAutoencoderConfig(img_size=img_size, patch_size=4,embed_dim=128, z_dim=16,depth_enc=4, depth_dec=4, num_heads=4),
        "big-z32":   ViTAutoencoderConfig(img_size=img_size, patch_size=4,embed_dim=192, z_dim=32,depth_enc=6, depth_dec=6, num_heads=6),
        # esempio con compressione in bit reale (abilita bpp/bit_ratio):
        "base-z16-q6": ViTAutoencoderConfig(img_size=img_size, patch_size=4,embed_dim=128, z_dim=16,depth_enc=4, depth_dec=4, num_heads=4,quantize=True, z_bits=6),
    }

    print("Settings for the tested model: ")
    print("tiny-z8: img_size= ",img_size,", patch_size=4, embed_dim=96, z_dim=8, depth_enc=3, depth_dec=3, num_heads=4")
    print("base-z16: img_size= ",img_size,", patch_size=4, embed_dim=128, z_dim=16, depth_enc=4, depth_dec=4, num_heads=4")
    print("big-z32: img_size= ",img_size,", patch_size=4, embed_dim=192, z_dim=32, depth_enc=6, depth_dec=6, num_heads=6")
    print("base-z16-q6: img_size= ",img_size,", patch_size=4, embed_dim=128, z_dim=16, depth_enc=4, depth_dec=4, num_heads=4, quantize=True, z_bits=6")

    results = {}
    trained = {}                        # modelli allenati, tenuti in memoria per la visualizzazione
    for name, cfg in configs.items():
        print(f"\n===== modello: {name} =====")
        model = ViTAutoencoder(cfg)
        t_train = train_model(model, train_loader, device, epochs=epochs, log_every=0)
        metrics = evaluate_model(model, test_loader, device)
        metrics["train_s"] = t_train
        results[name] = metrics
        trained[name] = model
        # salva i pesi per riuso (vedi codec.py)
        torch.save({"state_dict": model.state_dict(), "cfg": cfg.__dict__},
                   f"ckpt_{name}.pt")

    # ---- baseline JPEG ----
    print("\n===== baseline JPEG =====")
    jpeg_pts = jpeg_rate_distortion(test_loader, device)

    # ---- tabella riassuntiva ----
    print("\n\n================ RISULTATI ================")
    header = f"{'modello':14s} {'par':>10s} {'PSNR':>7s} {'SSIM':>6s} " \
             f"{'enc ms':>8s} {'dec ms':>8s} {'rid.dim':>8s} {'bpp':>7s} {'train s':>8s}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        bpp = f"{m['bpp']:.3f}" if "bpp" in m else "  -  "
        print(f"{name:14s} {m['params']:10,d} {m['psnr']:7.2f} {m['ssim']:6.3f} "
              f"{m['enc_s_per_img']*1e3:8.3f} {m['dec_s_per_img']*1e3:8.3f} "
              f"{m['dim_reduction']:7.2f}x {bpp:>7s} {m['train_s']:8.1f}")

    print("\n[info] confronto con JPEG valido in bpp: usare i modelli '-q*'")
    print("[info] curva R-D JPEG (bpp, psnr):",
          [(round(p['bpp'], 3), round(p['psnr'], 1)) for p in jpeg_pts])

    # ---- visualizzazione qualitativa finale ----
    # scegliamo come riferimento il bpp di un modello quantizzato, se presente,
    # cosi' la quality JPEG mostrata e' a bitrate confrontabile.
    target_bpp = next((results[n]["bpp"] for n in results if "bpp" in results[n]), None)
    print("\n===== visualizzazione ricostruzioni =====")
    visualize_reconstructions(trained, test_loader, device,
                              n_samples=n_viz, target_bpp=target_bpp)

    return results, jpeg_pts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar", choices=["cifar", "kodak"])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--n_viz", type=int, default=4, help="immagini da mostrare a fine run")
    args = ap.parse_args()
    run(dataset=args.dataset, epochs=args.epochs,
        batch_size=args.batch_size, img_size=args.img_size, n_viz=args.n_viz)
