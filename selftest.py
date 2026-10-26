"""
Headless self-test — verify OpenCV 5 and the agent pipeline work on every
configured feed, without the web UI. For judges / CI.

    python selftest.py            # test all feeds in feeds.json
    python selftest.py my.mp4     # test one video as a beach feed

Exits non-zero if OpenCV is not 5.x, a feed cannot be opened, or a beach feed
never produces a zone detection across the sampled frames.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
FRAMES = 2000  # sample up to this many frames per feed


def feeds():
    if len(sys.argv) > 1:
        return [{"id": "cli", "name": sys.argv[1], "kind": "beach", "video": sys.argv[1]}]
    cfg = HERE / "feeds.json"
    if cfg.exists():
        return json.loads(cfg.read_text())
    return [{"id": "beach", "kind": "beach", "video": "media/demo.mp4"},
            {"id": "sea", "kind": "sea", "video": "media/demo2.mp4"}]


ZONE = [(0.0, 0.32), (0.33, 0.28), (0.48, 1.0), (0.0, 1.0)]


def test_beach(path, zone_frac):
    bg = cv2.createBackgroundSubtractorMOG2(400, 28, True)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False, "cannot open video"
    zone = None
    detections = 0
    best = 0.0
    n = 0
    while n < FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        h, w = f.shape[:2]
        if zone is None:
            zone = np.array([(int(x * w), int(y * h)) for x, y in (zone_frac or ZONE)], np.int32)
        fg = bg.apply(cv2.GaussianBlur(f, (5, 5), 0))
        fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1]
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
        fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, k, iterations=2)
        cnts = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for c in cnts:
            a = cv2.contourArea(c) / (w * h)
            if a < 0.0006:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if cv2.pointPolygonTest(zone, (float(x + bw / 2), float(y + bh)), False) >= 0:
                roi = f[max(0, y):y + bh, max(0, x):x + bw]
                if roi.size:
                    ed = float(cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 60, 160).mean()) / 255.0
                    asp = bh / max(bw, 1)
                    up = 1.0 if 1.1 <= asp <= 4.0 else 0.4
                    conf = min(1.0, a / 0.0018) * (0.5 + ed) * up
                    best = max(best, conf)
                    if conf >= 0.55:
                        detections += 1
    cap.release()
    ok = detections > 0
    return ok, f"{n} frames · {detections} zone detections ≥0.55 · peak conf {best:.2f}"


def test_sea(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False, "cannot open video"
    prev = None
    ema = 8.0
    peak_rip = 0
    n = 0
    while n < FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        n += 1
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        motion = float(cv2.absdiff(g, prev).mean()) if prev is not None else 0.0
        prev = g
        foam = cv2.threshold(g, 170, 255, cv2.THRESH_BINARY)[1]
        cover = float(foam[int(f.shape[0] * 0.45):, :].mean()) / 255.0
        raw = 100.0 * (0.5 * min(1.0, motion / 2.0) + 0.5 * max(0.0, min(1.0, (cover - 0.05) / 0.05)))
        ema = 0.8 * ema + 0.2 * raw
        peak_rip = max(peak_rip, int(ema))
    cap.release()
    # a sea feed should at least reach a "moderate" surf index so the posture can elevate
    return peak_rip >= 35, f"{n} frames · peak surf/rip index {peak_rip}/100"


def main():
    print(f"OpenCV version: {cv2.__version__}")
    if not cv2.__version__.startswith("5."):
        print("FAIL: OpenCV 5.x is required.")
        sys.exit(1)
    rc = 0
    for fc in feeds():
        path = fc["video"]
        path = path if Path(path).is_absolute() else str(HERE / path)
        if not Path(path).exists():
            print(f"[{fc['id']}] FAIL: video not found: {path}")
            rc = 1
            continue
        if fc.get("kind", "beach") == "sea":
            ok, msg = test_sea(path)
        else:
            ok, msg = test_beach(path, fc.get("zone_frac"))
        print(f"[{fc['id']}] {'PASS' if ok else 'FAIL'}: {msg}")
        rc = rc or (0 if ok else 1)
    print("ALL PASS" if rc == 0 else "SELFTEST FAILED")
    sys.exit(rc)


if __name__ == "__main__":
    main()
