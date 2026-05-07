"""
Tests for watermark_qim.py
Run with: python -m pytest tests/ -v
"""
import sys
import os
import numpy as np

# Make sure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from watermark_qim import (
    generate_watermark,
    dct2_blocks,
    idct2_blocks,
    qim_embed_bit,
    qim_decode_bit,
    embed_watermark,
    extract_watermark,
    select_coefficients,
    reconstruct_image,
    compute_psnr,
    compute_ber,
    BLOCK_SIZE,
)


# ── Test 1 : watermark generation ─────────────────────────────────────────────
def test_generate_watermark_length():
    wm = generate_watermark(64, key=42)
    assert len(wm) == 64, "Watermark should have exactly 64 bits"


def test_generate_watermark_binary():
    wm = generate_watermark(64, key=42)
    assert all(b in [0, 1] for b in wm), "All watermark bits must be 0 or 1"


def test_generate_watermark_reproducible():
    wm1 = generate_watermark(64, key=42)
    wm2 = generate_watermark(64, key=42)
    assert np.array_equal(wm1, wm2), "Same key must produce same watermark"


def test_generate_watermark_different_keys():
    wm1 = generate_watermark(64, key=42)
    wm2 = generate_watermark(64, key=99)
    assert not np.array_equal(wm1, wm2), "Different keys should produce different watermarks"


# ── Test 2 : DCT / IDCT roundtrip ─────────────────────────────────────────────
def test_dct_idct_roundtrip():
    rng = np.random.default_rng(seed=0)
    img = (rng.random((64, 64)) * 255).astype(np.float64)
    dct = dct2_blocks(img)
    reconstructed = idct2_blocks(dct)
    assert np.allclose(img, reconstructed, atol=1e-6), "DCT → IDCT must be lossless"


# ── Test 3 : QIM embed / decode ───────────────────────────────────────────────
def test_qim_embed_decode_bit0():
    delta = 30.0
    embedded = qim_embed_bit(100.0, bit=0, delta=delta)
    decoded  = qim_decode_bit(embedded, delta=delta)
    assert decoded == 0, "Bit 0 must be decoded correctly after embedding"


def test_qim_embed_decode_bit1():
    delta = 30.0
    embedded = qim_embed_bit(100.0, bit=1, delta=delta)
    decoded  = qim_decode_bit(embedded, delta=delta)
    assert decoded == 1, "Bit 1 must be decoded correctly after embedding"


# ── Test 4 : full embed → extract pipeline (BER = 0 without attack) ──────────
def test_embed_extract_no_attack():
    # Build a synthetic 64×64 image
    y, x = np.mgrid[0:64, 0:64]
    img = ((np.sin(y / 10.0) * np.cos(x / 10.0) + 1) * 127).astype(np.uint8)

    key   = 42
    delta = 30.0
    bits  = 16

    wm_bits   = generate_watermark(bits, key)
    dct_orig  = dct2_blocks(img)
    positions = select_coefficients(64, 64, bits, key)
    dct_wm    = embed_watermark(dct_orig, wm_bits, positions, delta)
    img_wm    = reconstruct_image(dct_wm)

    extracted      = extract_watermark(dct2_blocks(img_wm), positions, delta)
    ber, n_errors  = compute_ber(wm_bits, extracted)

    assert ber == 0.0, f"BER without attack must be 0.0, got {ber} ({n_errors} errors)"


# ── Test 5 : PSNR is reasonable ───────────────────────────────────────────────
def test_psnr_above_threshold():
    y, x = np.mgrid[0:64, 0:64]
    img = ((np.sin(y / 10.0) * np.cos(x / 10.0) + 1) * 127).astype(np.uint8)

    wm_bits   = generate_watermark(16, 42)
    dct_orig  = dct2_blocks(img)
    positions = select_coefficients(64, 64, 16, 42)
    dct_wm    = embed_watermark(dct_orig, wm_bits, positions, delta=30.0)
    img_wm    = reconstruct_image(dct_wm)

    psnr = compute_psnr(img, img_wm)
    assert psnr > 30.0, f"PSNR should be > 30 dB, got {psnr:.2f} dB"


# ── Test 6 : Flask health endpoint ───────────────────────────────────────────
def test_health_endpoint():
    import importlib.util, types
    # Only test if app.py exists alongside
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app.py')
    if not os.path.exists(app_path):
        return  # skip if app.py not present

    os.environ['MPLBACKEND'] = 'Agg'
    spec = importlib.util.spec_from_file_location("app", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    client = module.app.test_client()
    response = client.get('/health')
    assert response.status_code == 200
    assert response.data == b"OK"
