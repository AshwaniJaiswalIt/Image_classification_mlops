"""
Model Performance Monitor – M5 Task 2
Sends a batch of simulated requests to the API, collects predictions,
compares against simulated true labels, and prints a performance report.

Usage:
    # Make sure the API is running first:
    #   docker run -d -p 8000:8000 --name cats-api cats-dogs-api:latest
    python model_monitor.py
"""
import json
import struct
import sys
import time
import urllib.request
import zlib
from datetime import datetime

BASE_URL = "http://localhost:8000"

# Simulated ground-truth labels (0=cat, 1=dog)
# In production these would come from human feedback or a labelling pipeline
SIMULATED_LABELS = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # 5 cats, 5 dogs


def make_png(rgb=(128, 128, 128)):
    """Create a minimal 1×1 PNG with the given RGB colour (no Pillow needed)."""
    def chunk(name, data):
        crc = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    r, g, b = rgb
    idat = zlib.compress(bytes([0, r, g, b]))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


# Colour palette: cat images are blue-ish, dog images are orange-ish (simulated)
SAMPLE_COLOURS = [
    (50, 100, 200),   # cat 0
    (60, 110, 210),   # cat 1
    (40,  90, 190),   # cat 2
    (55, 105, 205),   # cat 3
    (45,  95, 195),   # cat 4
    (200, 130,  50),  # dog 0
    (210, 140,  60),  # dog 1
    (190, 120,  40),  # dog 2
    (205, 135,  55),  # dog 3
    (195, 125,  45),  # dog 4
]


def predict(png_bytes):
    """POST a PNG to /predict and return the JSON response."""
    boundary = b"monitor_boundary"
    body = (
        b"--" + boundary + b"\r\n"
        b'Content-Disposition: form-data; name="file"; filename="img.png"\r\n'
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
        return json.loads(resp.read().decode())


def get_api_metrics():
    """Fetch /metrics from the API."""
    with urllib.request.urlopen(f"{BASE_URL}/metrics", timeout=10) as resp:
        return json.loads(resp.read().decode())


def main():
    print("=" * 60)
    print("MODEL PERFORMANCE MONITOR")
    print(f"Timestamp : {datetime.now().isoformat()}")
    print(f"API       : {BASE_URL}")
    print("=" * 60)

    # ── Health check ──────────────────────────────────────────────
    print("\n[1/3] Health check ...")
    try:
        with urllib.request.urlopen(f"{BASE_URL}/health", timeout=10) as resp:
            health = json.loads(resp.read().decode())
            if health.get("status") != "healthy":
                print(f"  FAIL: API unhealthy — {health}")
                sys.exit(1)
            print(f"  OK   — {health}")
    except Exception as e:
        print(f"  FAIL — Cannot reach API: {e}")
        sys.exit(1)

    # ── Batch prediction ──────────────────────────────────────────
    print(f"\n[2/3] Running {len(SIMULATED_LABELS)} simulated predictions ...")
    predictions = []
    latencies = []

    for idx, (colour, true_label) in enumerate(zip(SAMPLE_COLOURS, SIMULATED_LABELS)):
        png = make_png(colour)
        t0 = time.time()
        try:
            result = predict(png)
            latency = time.time() - t0
            pred_label = result["prediction"]
            predictions.append(pred_label)
            latencies.append(latency)
            status = "✓" if pred_label == true_label else "✗"
            print(f"  [{idx+1:2d}] true={true_label} pred={pred_label} "
                  f"({result['prediction_label']:3s}) conf={result['confidence']:.2f} "
                  f"latency={latency:.3f}s  {status}")
        except Exception as e:
            print(f"  [{idx+1:2d}] ERROR: {e}")
            predictions.append(-1)
            latencies.append(0.0)

    # ── Performance report ────────────────────────────────────────
    print("\n[3/3] Performance report ...")
    valid = [(p, t) for p, t in zip(predictions, SIMULATED_LABELS) if p != -1]
    if valid:
        correct = sum(p == t for p, t in valid)
        accuracy = correct / len(valid)
        avg_latency = sum(latencies) / len(latencies)
        print(f"  Samples     : {len(valid)}")
        print(f"  Correct     : {correct}")
        print(f"  Accuracy    : {accuracy:.2%}")
        print(f"  Avg latency : {avg_latency:.3f}s")

    # ── API-level metrics from /metrics endpoint ──────────────────
    print("\n  API Metrics (from /metrics endpoint):")
    try:
        api_metrics = get_api_metrics()
        print(f"    Total requests    : {api_metrics['total_requests']}")
        print(f"    Avg response time : {api_metrics['average_response_time_seconds']}s")
        print(f"    Success rate      : {api_metrics['success_rate']}")
    except Exception as e:
        print(f"    Could not fetch API metrics: {e}")

    # Save report to JSON
    report = {
        "timestamp": datetime.now().isoformat(),
        "samples": len(valid),
        "accuracy": round(accuracy, 4) if valid else None,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 4) if latencies else None,
        "predictions": predictions,
        "true_labels": SIMULATED_LABELS,
    }
    with open("monitoring_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n  Report saved to monitoring_report.json")
    print("\n" + "=" * 60)
    print("Monitoring complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
