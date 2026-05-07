"""
QIM Watermarking — Flask Web App
Wraps watermark_qim.py so it can run as a web server (Docker / Azure).
"""
import io
import base64
import os

import matplotlib
matplotlib.use('Agg')          # headless backend — REQUIRED in Docker/Azure (no screen)
import matplotlib.pyplot as plt
import numpy as np
import cv2
from flask import Flask, render_template_string, request

from watermark_qim import (
    generate_watermark,
    dct2_blocks,
    select_coefficients,
    embed_watermark,
    extract_watermark,
    reconstruct_image,
    compute_psnr,
    compute_ber,
    attack_jpeg,
    attack_gaussian,
    BLOCK_SIZE,
)

app = Flask(__name__)

# ─── HTML template (single-file, no external CSS) ────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QIM Digital Watermarking</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f23; color: #e0e0e0; padding: 30px; }
  h1 { color: #7c83fd; font-size: 1.8rem; margin-bottom: 6px; }
  .subtitle { color: #888; margin-bottom: 24px; font-size: 0.9rem; }
  .card { background: #1a1a3e; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
  .form-row { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-end; }
  label { font-size: 0.85rem; color: #aaa; display: flex; flex-direction: column; gap: 4px; }
  input, select { padding: 8px 10px; background: #0f0f23; color: #e0e0e0; border: 1px solid #444; border-radius: 6px; font-size: 0.9rem; width: 120px; }
  button { padding: 9px 22px; background: #7c83fd; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 0.95rem; font-weight: 600; }
  button:hover { background: #5a63e8; }
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
  .stat-box { background: #0f0f23; border-radius: 8px; padding: 14px; text-align: center; }
  .stat-value { font-size: 1.4rem; font-weight: bold; color: #7c83fd; }
  .stat-label { font-size: 0.75rem; color: #888; margin-top: 4px; }
  .images { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
  .img-card { background: #1a1a3e; border-radius: 10px; overflow: hidden; }
  .img-card h3 { padding: 12px 16px; font-size: 0.9rem; color: #aaa; border-bottom: 1px solid #333; }
  .img-card img { width: 100%; display: block; }
  .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; margin-top: 8px; }
  .good { background: #1a3a2a; color: #4caf87; }
  .warn { background: #3a2a1a; color: #e08050; }
</style>
</head>
<body>

<h1>🔒 QIM Digital Watermarking</h1>
<p class="subtitle">DCT-domain watermarking with robustness evaluation — CI/CD deployed on Azure</p>

<div class="card">
  <form method="POST">
    <div class="form-row">
      <label>Secret Key
        <input type="number" name="key" value="{{ key }}">
      </label>
      <label>Delta (Δ)
        <input type="number" name="delta" value="{{ delta }}" min="1" max="100">
      </label>
      <label>Bits
        <input type="number" name="bits" value="{{ bits }}" min="8" max="512">
      </label>
      <label>Attack
        <select name="attack">
          <option value="jpeg"     {% if attack=='jpeg'     %}selected{% endif %}>JPEG Compression</option>
          <option value="gaussian" {% if attack=='gaussian' %}selected{% endif %}>Gaussian Noise</option>
          <option value="none"     {% if attack=='none'     %}selected{% endif %}>No Attack</option>
        </select>
      </label>
      <button type="submit">▶ Run</button>
    </div>
  </form>
</div>

{% if psnr %}
<div class="card">
  <div class="stats-grid">
    <div class="stat-box">
      <div class="stat-value">{{ psnr }} dB</div>
      <div class="stat-label">PSNR (quality)</div>
      <span class="badge {{ 'good' if psnr_float > 35 else 'warn' }}">
        {{ '✓ Invisible' if psnr_float > 35 else '⚠ Visible' }}
      </span>
    </div>
    <div class="stat-box">
      <div class="stat-value">{{ ber_clean }}</div>
      <div class="stat-label">BER without attack</div>
      <span class="badge {{ 'good' if ber_clean == '0.0000' else 'warn' }}">
        {{ '✓ Perfect' if ber_clean == '0.0000' else '⚠ Errors' }}
      </span>
    </div>
    <div class="stat-box">
      <div class="stat-value">{{ ber_attacked }}</div>
      <div class="stat-label">BER after attack</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">{{ bits }}</div>
      <div class="stat-label">Watermark bits</div>
    </div>
    <div class="stat-box">
      <div class="stat-value">{{ attack_label }}</div>
      <div class="stat-label">Attack applied</div>
    </div>
  </div>
</div>

<div class="images">
  <div class="img-card">
    <h3>🖼 Original Image</h3>
    <img src="data:image/png;base64,{{ img_orig_b64 }}">
  </div>
  <div class="img-card">
    <h3>💧 Watermarked Image — PSNR {{ psnr }} dB</h3>
    <img src="data:image/png;base64,{{ img_wm_b64 }}">
  </div>
  <div class="img-card">
    <h3>⚔ After Attack: {{ attack_label }}</h3>
    <img src="data:image/png;base64,{{ img_attacked_b64 }}">
  </div>
</div>
{% endif %}

</body>
</html>
"""


def array_to_b64(img_array: np.ndarray) -> str:
    """Convert a grayscale numpy uint8 array to a base64 PNG string."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_array, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02, dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def run_pipeline(key: int, delta: float, bits: int, attack: str):
    """Run the full QIM watermarking pipeline and return results."""
    # ── Build the host image ──────────────────────────────────────────────────
    # Try to load lena.png (or any image placed next to app.py).
    # If not found, generate the smooth synthetic image from watermark_qim.py.
    image_path = os.path.join(os.path.dirname(__file__), 'lena.png')
    if os.path.exists(image_path):
        img_orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        y_idx, x_idx = np.mgrid[0:512, 0:512]
        img_orig = ((np.sin(y_idx / 30.0) * np.cos(x_idx / 30.0) + 1) * 127).astype(np.uint8)
        img_orig = cv2.GaussianBlur(img_orig, (15, 15), 4)

    # Crop to multiple of BLOCK_SIZE
    h, w = img_orig.shape
    h = (h // BLOCK_SIZE) * BLOCK_SIZE
    w = (w // BLOCK_SIZE) * BLOCK_SIZE
    img_orig = img_orig[:h, :w]

    # ── Watermark pipeline ────────────────────────────────────────────────────
    wm_bits   = generate_watermark(bits, key)
    dct_orig  = dct2_blocks(img_orig)
    positions = select_coefficients(h, w, bits, key)
    dct_wm    = embed_watermark(dct_orig, wm_bits, positions, delta)
    img_wm    = reconstruct_image(dct_wm)

    psnr_val     = compute_psnr(img_orig, img_wm)
    ext_clean    = extract_watermark(dct2_blocks(img_wm), positions, delta)
    ber_clean, _ = compute_ber(wm_bits, ext_clean)

    # ── Attack ────────────────────────────────────────────────────────────────
    if attack == 'gaussian':
        img_attacked = attack_gaussian(img_wm)
        attack_label = 'Gaussian noise (σ=5)'
    elif attack == 'jpeg':
        img_attacked = attack_jpeg(img_wm, quality=50)
        attack_label = 'JPEG compression (Q=50)'
    else:
        img_attacked = img_wm.copy()
        attack_label = 'None'

    ext_attacked    = extract_watermark(dct2_blocks(img_attacked), positions, delta)
    ber_attacked, _ = compute_ber(wm_bits, ext_attacked)

    return img_orig, img_wm, img_attacked, psnr_val, ber_clean, ber_attacked, attack_label


@app.route('/', methods=['GET', 'POST'])
def index():
    key    = int(request.form.get('key',   42))
    delta  = float(request.form.get('delta', 30))
    bits   = int(request.form.get('bits',  64))
    attack = request.form.get('attack', 'jpeg')

    ctx = dict(key=key, delta=int(delta), bits=bits, attack=attack,
               psnr=None, ber_clean=None, ber_attacked=None,
               attack_label=None, img_orig_b64=None,
               img_wm_b64=None, img_attacked_b64=None, psnr_float=0)

    if request.method == 'POST':
        img_orig, img_wm, img_attacked, psnr_val, ber_clean, ber_attacked, attack_label = \
            run_pipeline(key, delta, bits, attack)

        ctx.update(
            psnr=f"{psnr_val:.2f}",
            psnr_float=psnr_val,
            ber_clean=f"{ber_clean:.4f}",
            ber_attacked=f"{ber_attacked:.4f}",
            attack_label=attack_label,
            img_orig_b64=array_to_b64(img_orig),
            img_wm_b64=array_to_b64(img_wm),
            img_attacked_b64=array_to_b64(img_attacked),
        )

    return render_template_string(HTML, **ctx)


@app.route('/health')
def health():
    """Health-check endpoint used by Azure App Service."""
    return "OK", 200


if __name__ == '__main__':
    # Local run: python app.py
    app.run(host='0.0.0.0', port=5000, debug=False)
