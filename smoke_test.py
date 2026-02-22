#!/usr/bin/env python3
"""
Smoke test script for the Cats vs Dogs API.
Run after deployment to verify the service is healthy and predictions work.
Exits with code 1 if any check fails (fails the CI/CD pipeline).
"""
import sys
import time
import urllib.request
import urllib.error
import json
import struct
import zlib

BASE_URL = "http://localhost:8000"
MAX_RETRIES = 10
RETRY_DELAY = 10  # seconds


def wait_for_service():
    """Wait for the service to become available."""
    print(f"Waiting for service at {BASE_URL} ...")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=5) as resp:
                if resp.status == 200:
                    print(f"  Service is up (attempt {attempt})")
                    return True
        except Exception:
            pass
        print(f"  Attempt {attempt}/{MAX_RETRIES} failed, retrying in {RETRY_DELAY}s ...")
        time.sleep(RETRY_DELAY)
    return False


def test_health():
    """Test: GET /health returns healthy status."""
    print("\n[1/3] Smoke test: /health")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=10) as resp:
            data = json.loads(resp.read().decode())
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            assert data.get("status") == "healthy", f"Expected 'healthy', got {data}"
            assert data.get("models_loaded") is True, "Model not loaded"
            print(f"  PASS — {data}")
    except Exception as e:
        print(f"  FAIL — {e}")
        sys.exit(1)


def test_model_info():
    """Test: GET /model/info returns correct model metadata."""
    print("\n[2/3] Smoke test: /model/info")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/model/info", timeout=10) as resp:
            data = json.loads(resp.read().decode())
            assert resp.status == 200
            assert "model_type" in data
            assert "classes" in data
            print(f"  PASS — model_type={data['model_type']}, classes={data['classes']}")
    except Exception as e:
        print(f"  FAIL — {e}")
        sys.exit(1)


def make_minimal_png():
    """Create a minimal valid 1x1 white PNG image in memory (no Pillow needed)."""
    def chunk(name, data):
        c = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1 RGB
    idat_data = zlib.compress(b"\x00\xFF\xFF\xFF")         # filter byte + white pixel
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", idat_data)
        + chunk(b"IEND", b"")
    )
    return png


def test_predict():
    """Test: POST /predict with a dummy image returns a valid prediction."""
    print("\n[3/3] Smoke test: /predict")
    try:
        png_bytes = make_minimal_png()
        boundary = b"smoketest_boundary"
        body = (
            b"--" + boundary + b"\r\n"
            b'Content-Disposition: form-data; name="file"; filename="test.png"\r\n'
            b"Content-Type: image/png\r\n\r\n"
            + png_bytes
            + b"\r\n--" + boundary + b"--\r\n"
        )
        req = urllib.request.Request(
            f"{BASE_URL}/predict",
            data=body,
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            assert resp.status == 200
            assert "prediction_label" in data
            assert data["prediction_label"] in ("cat", "dog")
            assert "confidence" in data
            print(f"  PASS — label={data['prediction_label']}, confidence={data['confidence']}")
    except Exception as e:
        print(f"  FAIL — {e}")
        sys.exit(1)


if __name__ == "__main__":
    if not wait_for_service():
        print("\nFAIL — Service did not become available in time.")
        sys.exit(1)

    test_health()
    test_model_info()
    test_predict()

    print("\n✅ All smoke tests passed.")
    sys.exit(0)
