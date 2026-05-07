"""
Mini-Projet : Sécurisation des images 2D par tatouage numérique basé sur QIM
============================================================================
Conception et implémentation d'un système de tatouage numérique robuste
basé sur la méthode QIM (Quantization Index Modulation) en Python.

Librairies : opencv-python  numpy  matplotlib  scikit-image  scipy

Utilisation :
    python watermark_qim.py --image lena.png --key 42 --delta 30 --bits 64 --attack jpeg
    python watermark_qim.py --image lena.png --key 42 --delta 30 --bits 64 --attack gaussian
    python watermark_qim.py --image lena.png --key 42 --delta 30 --bits 64 --attack none
    python watermark_qim.py          # image synthétique par défaut
"""

import argparse
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct, idct
from skimage.metrics import peak_signal_noise_ratio as psnr_func

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────
BLOCK_SIZE = 8          # Taille des blocs DCT (standard JPEG)

# Positions de fréquence moyenne dans un bloc 8×8 (zigzag, hors DC)
# (ligne, colonne) dans le bloc local — évite les très basses (DC, AC1)
# et les très hautes fréquences (trop fragiles).
MID_FREQ_POSITIONS = [
    (1, 2), (1, 3), (2, 1), (2, 2), (2, 3),
    (3, 1), (3, 2), (3, 3), (3, 4),
    (4, 1), (4, 2), (4, 3),
    (5, 1), (5, 2),
]  # 14 positions par bloc


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 — LECTURE DE L'IMAGE HÔTE (fonction utilitaire : watermark binaire)
# ─────────────────────────────────────────────────────────────────────────────
def generate_watermark(n_bits: int, key: int) -> np.ndarray:
    """
    Génère un watermark binaire pseudo-aléatoire reproductible.

    Paramètres
    ----------
    n_bits : int
        Nombre de bits à générer.
    key : int
        Clé secrète utilisée comme graine du générateur pseudo-aléatoire.

    Retour
    ------
    np.ndarray
        Tableau d'entiers {0, 1} de longueur n_bits.
    """
    rng = np.random.default_rng(seed=key)
    return rng.integers(0, 2, size=n_bits, dtype=np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 — TRANSFORMATION FRÉQUENTIELLE (DCT 2D PAR BLOCS 8×8)
# ─────────────────────────────────────────────────────────────────────────────
def dct2_blocks(img: np.ndarray) -> np.ndarray:
    """
    Applique la DCT 2D normalisée ('ortho') sur chaque bloc 8×8 de l'image.

    La transformation est séparable : DCT sur les colonnes (axis=0) puis
    sur les lignes (axis=1), en utilisant scipy.fft.dct.
    Version vectorisée : tous les blocs sont traités simultanément.

    Paramètres
    ----------
    img : np.ndarray
        Image en niveaux de gris (dimensions multiples de BLOCK_SIZE).

    Retour
    ------
    np.ndarray
        Matrice de coefficients DCT (float64), même dimensions que img.
    """
    h, w = img.shape
    # Reshape en (nb_blocs_v, 8, nb_blocs_h, 8) puis (N, 8, 8)
    blocks = img.reshape(h // BLOCK_SIZE, BLOCK_SIZE,
                         w // BLOCK_SIZE, BLOCK_SIZE)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, BLOCK_SIZE, BLOCK_SIZE)
    blocks = blocks.astype(np.float64)
    # DCT 2D = DCT sur colonnes (axis=1) puis sur lignes (axis=2)
    blocks = dct(blocks, axis=1, norm='ortho')
    blocks = dct(blocks, axis=2, norm='ortho')
    # Remettre en forme (h, w)
    nb_v, nb_h = h // BLOCK_SIZE, w // BLOCK_SIZE
    out = blocks.reshape(nb_v, nb_h, BLOCK_SIZE, BLOCK_SIZE)
    out = out.transpose(0, 2, 1, 3).reshape(h, w)
    return out


def idct2_blocks(dct_img: np.ndarray) -> np.ndarray:
    """
    Applique l'IDCT 2D normalisée ('ortho') sur chaque bloc 8×8.

    Ordre inverse de dct2_blocks : IDCT sur les lignes puis les colonnes.
    Version vectorisée : tous les blocs sont traités simultanément.

    Paramètres
    ----------
    dct_img : np.ndarray
        Matrice de coefficients DCT (float64).

    Retour
    ------
    np.ndarray
        Image reconstruite en float64 (non clippée).
    """
    h, w = dct_img.shape
    nb_v, nb_h = h // BLOCK_SIZE, w // BLOCK_SIZE
    blocks = dct_img.reshape(nb_v, BLOCK_SIZE, nb_h, BLOCK_SIZE)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, BLOCK_SIZE, BLOCK_SIZE)
    blocks = blocks.copy()
    # IDCT 2D = IDCT sur lignes (axis=2) puis colonnes (axis=1)
    blocks = idct(blocks, axis=2, norm='ortho')
    blocks = idct(blocks, axis=1, norm='ortho')
    out = blocks.reshape(nb_v, nb_h, BLOCK_SIZE, BLOCK_SIZE)
    out = out.transpose(0, 2, 1, 3).reshape(h, w)
    return out


def select_coefficients(h: int, w: int, n_bits: int, key: int) -> list:
    """
    Sélectionne pseudo-aléatoirement n_bits coefficients DCT de fréquence
    moyenne en utilisant la clé secrète.

    La graine est dérivée de la clé (key ^ 0xDEADBEEF) pour être
    statistiquement indépendante de la graine du watermark.

    Paramètres
    ----------
    h : int
        Hauteur de l'image (multiple de BLOCK_SIZE).
    w : int
        Largeur de l'image (multiple de BLOCK_SIZE).
    n_bits : int
        Nombre de positions à sélectionner.
    key : int
        Clé secrète entière.

    Retour
    ------
    list[tuple[int, int]]
        Liste de tuples (row_global, col_global) dans l'image DCT.

    Raises
    ------
    ValueError
        Si n_bits dépasse le nombre de positions candidates disponibles.
    """
    candidates = []
    for bi in range(h // BLOCK_SIZE):
        for bj in range(w // BLOCK_SIZE):
            for (li, lj) in MID_FREQ_POSITIONS:
                candidates.append((bi * BLOCK_SIZE + li, bj * BLOCK_SIZE + lj))

    if n_bits > len(candidates):
        raise ValueError(
            f"Trop de bits demandés ({n_bits}) pour les blocs disponibles "
            f"({len(candidates)} positions candidates). "
            f"Réduisez --bits ou utilisez une image plus grande."
        )

    # Graine différente de celle du watermark pour indépendance statistique
    rng = np.random.default_rng(seed=key ^ 0xDEADBEEF)
    indices = rng.choice(len(candidates), size=n_bits, replace=False)
    return [candidates[i] for i in indices]


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 3 — INSERTION DU WATERMARK PAR QIM
# ─────────────────────────────────────────────────────────────────────────────
def qim_embed_bit(x: float, bit: int, delta: float) -> float:
    """
    Quantification QIM scalaire pour un bit.

    Le domaine réel est partitionné en deux réseaux de quantification :
      Λ_0 = { k·Δ           : k ∈ ℤ }   → encode le bit 0
      Λ_1 = { k·Δ + Δ/2     : k ∈ ℤ }   → encode le bit 1

    Paramètres
    ----------
    x : float
        Coefficient DCT original.
    bit : int
        Bit à insérer (0 ou 1).
    delta : float
        Pas de quantification Δ (doit être > 0).

    Retour
    ------
    float
        Coefficient DCT modifié (quantifié vers le réseau Λ_bit).
    """
    if bit == 0:
        return delta * round(x / delta)
    else:
        return delta * round((x - delta / 2.0) / delta) + delta / 2.0


def qim_decode_bit(y: float, delta: float) -> int:
    """
    Décodage QIM aveugle (sans image originale).

    Compare la distance de y aux deux réseaux Λ_0 et Λ_1 et choisit
    le plus proche :
        d0 = |y − Δ·round(y/Δ)|
        d1 = |y − (Δ·round((y−Δ/2)/Δ) + Δ/2)|
        bit = 0 si d0 ≤ d1, sinon 1

    Paramètres
    ----------
    y : float
        Coefficient DCT (possiblement bruité).
    delta : float
        Pas de quantification Δ.

    Retour
    ------
    int
        Bit décodé (0 ou 1).
    """
    q0 = delta * round(y / delta)
    q1 = delta * round((y - delta / 2.0) / delta) + delta / 2.0
    return 0 if abs(y - q0) <= abs(y - q1) else 1


def embed_watermark(dct_img: np.ndarray,
                    wm_bits: np.ndarray,
                    positions: list,
                    delta: float) -> np.ndarray:
    """
    Insère les bits du watermark dans les coefficients DCT sélectionnés.

    Paramètres
    ----------
    dct_img : np.ndarray
        Matrice de coefficients DCT de l'image originale.
    wm_bits : np.ndarray
        Tableau binaire du watermark à insérer.
    positions : list[tuple[int, int]]
        Positions (row, col) des coefficients DCT à modifier.
    delta : float
        Pas de quantification QIM.

    Retour
    ------
    np.ndarray
        Matrice DCT avec le watermark inséré.
    """
    dct_wm = dct_img.copy()
    for idx, (r, c) in enumerate(positions):
        dct_wm[r, c] = qim_embed_bit(dct_wm[r, c], int(wm_bits[idx]), delta)
    return dct_wm


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 4 — SIMULATION D'ATTAQUES
# ─────────────────────────────────────────────────────────────────────────────
def attack_gaussian(img: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """
    Applique un bruit gaussien additif (AWGN) sur l'image.

    Paramètres
    ----------
    img : np.ndarray
        Image en niveaux de gris (uint8).
    sigma : float
        Écart-type du bruit gaussien.

    Retour
    ------
    np.ndarray
        Image bruitée (uint8, clippée dans [0, 255]).
    """
    rng = np.random.default_rng(seed=0)
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def attack_jpeg(img: np.ndarray, quality: int = 50) -> np.ndarray:
    """
    Applique une compression JPEG en mémoire via OpenCV (imencode/imdecode).
    Aucun fichier temporaire n'est créé sur le disque.

    Paramètres
    ----------
    img : np.ndarray
        Image en niveaux de gris (uint8).
    quality : int
        Qualité JPEG (1–100). Plus bas = plus de perte.

    Retour
    ------
    np.ndarray
        Image après compression/décompression JPEG (uint8).
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode('.jpg', img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 5 — EXTRACTION ET ÉVALUATION (métriques PSNR et BER)
# ─────────────────────────────────────────────────────────────────────────────
def extract_watermark(dct_img: np.ndarray,
                      positions: list,
                      delta: float) -> np.ndarray:
    """
    Extrait le watermark des coefficients DCT (mode aveugle — sans image originale).

    Paramètres
    ----------
    dct_img : np.ndarray
        Matrice de coefficients DCT de l'image (possiblement attaquée).
    positions : list[tuple[int, int]]
        Positions des coefficients DCT (même liste que lors de l'insertion).
    delta : float
        Pas de quantification QIM (même valeur que lors de l'insertion).

    Retour
    ------
    np.ndarray
        Tableau binaire {0, 1} du watermark extrait (même longueur que positions).
    """
    return np.array(
        [qim_decode_bit(dct_img[r, c], delta) for (r, c) in positions],
        dtype=np.int32
    )


def compute_psnr(img_ref: np.ndarray, img_test: np.ndarray) -> float:
    """
    Calcule le Peak Signal-to-Noise Ratio (dB) via scikit-image.
    Un PSNR > 35 dB indique une distorsion imperceptible à l'œil humain.

    Paramètres
    ----------
    img_ref : np.ndarray
        Image de référence (originale).
    img_test : np.ndarray
        Image à évaluer (tatouée ou attaquée).

    Retour
    ------
    float
        Valeur du PSNR en décibels.
    """
    return psnr_func(img_ref, img_test, data_range=255)


def compute_ber(original_bits: np.ndarray,
                extracted_bits: np.ndarray) -> tuple:
    """
    Calcule le Bit Error Rate et le nombre absolu d'erreurs.
    BER = 0.0 → extraction parfaite ; BER = 0.5 → aléatoire (pire cas).

    Paramètres
    ----------
    original_bits : np.ndarray
        Watermark original (binaire).
    extracted_bits : np.ndarray
        Watermark extrait (binaire).

    Retour
    ------
    tuple[float, int]
        (ber, n_erreurs) — taux d'erreur et nombre absolu d'erreurs.
    """
    n_errors = int(np.sum(original_bits != extracted_bits))
    ber = n_errors / len(original_bits)
    return ber, n_errors


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 6 — RECONSTRUCTION DE L'IMAGE TATOUÉE + AFFICHAGE
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct_image(dct_wm: np.ndarray) -> np.ndarray:
    """
    Reconstruit l'image tatouée depuis le domaine DCT par IDCT 2D,
    puis clippe dans [0, 255] et convertit en uint8.

    Paramètres
    ----------
    dct_wm : np.ndarray
        Matrice de coefficients DCT (avec watermark inséré).

    Retour
    ------
    np.ndarray
        Image reconstruite en uint8.
    """
    img = idct2_blocks(dct_wm)
    return np.clip(img, 0, 255).astype(np.uint8)


def display_results(img_orig, img_wm, img_attacked,
                    psnr_wm, ber_clean, ber_attacked,
                    n_err_clean, n_err_attacked,
                    attack_label, n_bits, save_path='result_watermark.png'):
    """
    Affiche les trois images côte à côte et sauvegarde la figure en PNG.

    Paramètres
    ----------
    img_orig : np.ndarray
        Image originale.
    img_wm : np.ndarray
        Image tatouée.
    img_attacked : np.ndarray
        Image après attaque.
    psnr_wm : float
        PSNR entre image originale et tatouée.
    ber_clean : float
        BER sans attaque.
    ber_attacked : float
        BER après attaque.
    n_err_clean : int
        Nombre d'erreurs sans attaque.
    n_err_attacked : int
        Nombre d'erreurs après attaque.
    attack_label : str
        Description de l'attaque appliquée.
    n_bits : int
        Nombre total de bits du watermark.
    save_path : str
        Chemin de sauvegarde de la figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        'Système de tatouage numérique QIM — DCT 2D par blocs',
        fontsize=13, fontweight='bold'
    )

    axes[0].imshow(img_orig, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Image originale', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(img_wm, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title(
        f'Image tatouée\nPSNR = {psnr_wm:.2f} dB\n'
        f'BER = {ber_clean:.4f} ({n_err_clean}/{n_bits})',
        fontsize=10
    )
    axes[1].axis('off')

    axes[2].imshow(img_attacked, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(
        f'Après attaque : {attack_label}\n'
        f'BER = {ber_attacked:.4f} ({n_err_attacked}/{n_bits})',
        fontsize=10
    )
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    print(f"[INFO] Figure sauvegardee : '{save_path}'")
    plt.show(block=False)
    plt.pause(0.5)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE PRINCIPAL — ÉTAPES 1 À 6 (CLI)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Formateur combiné : aide par défaut + epilog formaté
    class Formatter(argparse.RawDescriptionHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
        pass

    parser = argparse.ArgumentParser(
        description='Tatouage numérique QIM sur image 2D (domaine DCT)',
        formatter_class=Formatter,
        epilog=(
            'Exemples :\n'
            '  python watermark_qim.py --image lena.png --attack jpeg\n'
            '  python watermark_qim.py --image lena.png --attack gaussian --gaussian-sigma 10\n'
            '  python watermark_qim.py --image lena.png --attack none\n'
            '  python watermark_qim.py   # image synthétique'
        )
    )
    parser.add_argument('--image', type=str, default=None,
                        help="Chemin vers l'image hôte (.png/.jpg). Omis → image synthétique.")
    parser.add_argument('--key', type=int, default=42,
                        help='Clé secrète entière (reproductibilité)')
    parser.add_argument('--delta', type=float, default=30.0,
                        help='Pas de quantification QIM Δ (↑ = plus robuste, moins invisible)')
    parser.add_argument('--bits', type=int, default=64,
                        help='Nombre de bits du watermark à insérer')
    parser.add_argument('--attack', type=str, default='jpeg',
                        choices=['gaussian', 'jpeg', 'none'],
                        help='Attaque à simuler')
    parser.add_argument('--jpeg-quality', type=int, default=50,
                        help="Qualité JPEG (1–100) pour l'attaque JPEG")
    parser.add_argument('--gaussian-sigma', type=float, default=5.0,
                        help='Écart-type σ du bruit gaussien')
    args = parser.parse_args()

    # Validation du pas de quantification
    if args.delta <= 0:
        print("[ERREUR] Le pas Delta (--delta) doit etre strictement positif.",
              file=sys.stderr)
        sys.exit(1)

    t_start = time.perf_counter()

    # ── ÉTAPE 1 : Lecture de l'image hôte ─────────────────────────────────────
    if args.image:
        img_orig = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if img_orig is None:
            print(f"[ERREUR] Impossible de lire '{args.image}'.", file=sys.stderr)
            sys.exit(1)
        print(f"[INFO] Image chargee : {args.image}")
    else:
        print("[INFO] Aucune image fournie -- generation d'une image synthetique (512x512).")
        # Gradient lisse + texture fine pour simuler une photo naturelle
        y_idx, x_idx = np.mgrid[0:512, 0:512]
        img_orig = ((np.sin(y_idx / 30.0) * np.cos(x_idx / 30.0) + 1) * 127).astype(np.uint8)
        img_orig = cv2.GaussianBlur(img_orig, (15, 15), 4)

    # Rogner à un multiple de BLOCK_SIZE
    h, w = img_orig.shape
    h = (h // BLOCK_SIZE) * BLOCK_SIZE
    w = (w // BLOCK_SIZE) * BLOCK_SIZE
    img_orig = img_orig[:h, :w]
    print(f"[INFO] Dimensions (apres rognage) : {h}x{w} pixels")

    # ── Génération du watermark binaire ───────────────────────────────────────
    wm_bits = generate_watermark(args.bits, args.key)
    print(f"[INFO] Watermark ({args.bits} bits, cle={args.key}) : "
          f"{wm_bits[:16].tolist()}{'...' if args.bits > 16 else ''}")

    # ── ÉTAPE 2 : Transformation fréquentielle (DCT 2D par blocs 8×8) ────────
    dct_orig = dct2_blocks(img_orig)

    # ── Sélection pseudo-aléatoire des coefficients de fréquence moyenne ─────
    positions = select_coefficients(h, w, args.bits, args.key)
    print(f"[INFO] {len(positions)} coefficients DCT selectionnes "
          f"(frequence moyenne, cle={args.key})")

    # ── ÉTAPE 3 : Insertion du watermark par QIM ──────────────────────────────
    dct_watermarked = embed_watermark(dct_orig, wm_bits, positions, args.delta)

    # ── ÉTAPE 6 : Reconstruction de l'image tatouée (IDCT) ───────────────────
    img_wm = reconstruct_image(dct_watermarked)

    # ── ÉTAPE 5a : Extraction et évaluation AVANT attaque ─────────────────────
    psnr_val = compute_psnr(img_orig, img_wm)
    dct_wm_clean = dct2_blocks(img_wm)
    extracted_clean = extract_watermark(dct_wm_clean, positions, args.delta)
    ber_clean, n_err_clean = compute_ber(wm_bits, extracted_clean)

    print(f"\n{'-'*55}")
    # PSNR > 35 dB → distorsion imperceptible à l'œil humain
    print(f"  PSNR (original vs tatouee) : {psnr_val:.2f} dB")
    print(f"  BER  (sans attaque)        : {ber_clean:.4f}  "
          f"({n_err_clean}/{args.bits} erreurs)")

    # ── ÉTAPE 4 : Simulation d'attaque ────────────────────────────────────────
    if args.attack == 'gaussian':
        img_attacked = attack_gaussian(img_wm, sigma=args.gaussian_sigma)
        attack_label = f'Bruit gaussien (sigma={args.gaussian_sigma})'
    elif args.attack == 'jpeg':
        img_attacked = attack_jpeg(img_wm, quality=args.jpeg_quality)
        attack_label = f'Compression JPEG (Q={args.jpeg_quality})'
    else:
        img_attacked = img_wm.copy()
        attack_label = 'Aucune'

    # ── ÉTAPE 5b : Extraction et évaluation APRÈS attaque ─────────────────────
    dct_attacked = dct2_blocks(img_attacked)
    extracted_attacked = extract_watermark(dct_attacked, positions, args.delta)
    ber_attacked, n_err_attacked = compute_ber(wm_bits, extracted_attacked)

    print(f"  BER  (apres {attack_label}) : {ber_attacked:.4f}  "
          f"({n_err_attacked}/{args.bits} erreurs)")

    # ── Sauvegarde des images ─────────────────────────────────────────────────
    cv2.imwrite('watermarked_image.png', img_wm)
    cv2.imwrite('attacked_image.png', img_attacked)
    print(f"\n[INFO] Images sauvegardees : 'watermarked_image.png', 'attacked_image.png'")

    # -- Affichage (matplotlib -- 3 images cote a cote) ------------------------
    display_results(
        img_orig, img_wm, img_attacked,
        psnr_val, ber_clean, ber_attacked,
        n_err_clean, n_err_attacked,
        attack_label, args.bits
    )

    # -- Temps d'execution -------------------------------------------------
    elapsed = time.perf_counter() - t_start
    print(f"\n{'-'*55}")
    print(f"  Temps d'execution : {elapsed:.3f} sec", end="")
    if elapsed > 5:
        print("  [!] Superieur a 5 sec (specification non-fonctionnelle)", end="")
    print()

    # -- Resume final ------------------------------------------------------
    print(f"\n{'='*55}")
    print("  RESUME")
    print(f"{'='*55}")
    print(f"  Image             : {h}x{w} pixels")
    print(f"  Cle secrete       : {args.key}")
    print(f"  Delta (quantif.)  : {args.delta}")
    print(f"  Bits du watermark : {args.bits}")
    print(f"  Attaque simulee   : {attack_label}")
    print(f"  PSNR              : {psnr_val:.2f} dB")
    print(f"  BER sans attaque  : {ber_clean:.4f}")
    print(f"  BER avec attaque  : {ber_attacked:.4f}")
    print(f"  Temps             : {elapsed:.3f} sec")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
