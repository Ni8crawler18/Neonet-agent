"""
Neonet Vision Agent — OpenCV AI Competition 2026 (Agentic Vision Award)

A two-feed agentic vision command centre built on OpenCV 5. An agent runs a
many-stage perception pipeline on each feed, FUSES the two feeds, and escalates
confirmed hazards to a human operator for approval.

  Feed A — Cove beach cam      : geofenced water zone, swimmer-in-zone detection
  Feed B — Offshore sea-state  : surf energy / whitewater -> sea state + rip risk

  Cross-feed reasoning: Feed B's sea state sets a RISK POSTURE that changes how
  the agent treats events on Feed A (a swimmer entering the water under a ROUGH
  sea with HIGH rip risk is escalated to CRITICAL, not advisory).

Every substantive operation — decode, grayscale, blur, MOG2, morphology,
contours, point-in-polygon, Canny, frame-differencing, thresholding, encoding —
runs on OpenCV 5, CPU-only. Each is timed and surfaced as a pipeline stage.

Views (single-page, sidebar nav):
  Overview · Beach feed · Sea-state feed · OpenCV pipeline · Cross-feed reasoning · Incidents
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

HERE = Path(__file__).parent
TARGET_FPS = 12
ZONE_FRAC = [(0.00, 0.32), (0.33, 0.28), (0.48, 1.00), (0.00, 1.00)]
MIN_AREA_FRAC = 0.0006
CONFIRM_FRAC = 0.0018
CONF_GATE = 0.55


def stage_timer():
    """Returns (record, stages) — record('name', fn) times fn and stores ms."""
    stages: dict[str, float] = {}

    def record(name, fn):
        t0 = time.perf_counter()
        r = fn()
        stages[name] = round((time.perf_counter() - t0) * 1000, 2)
        return r

    return record, stages


# --------------------------------------------------------------------------- #
# Shared cross-feed state
# --------------------------------------------------------------------------- #
@dataclass
class Posture:
    level: str = "CALM"          # CALM | ELEVATED | HIGH
    sea_state: str = "calm"      # calm | moderate | rough
    rip_risk: str = "low"        # low | moderate | high
    rip_score: int = 0           # 0..100
    whitewater: float = 0.0      # fraction


POSTURE = Posture()
FUSION_LOG: deque = deque(maxlen=30)
FUSION_LOCK = threading.Lock()


def fusion(msg: str) -> None:
    with FUSION_LOCK:
        FUSION_LOG.appendleft({"t": time.strftime("%H:%M:%S"), "detail": msg})


# Alerts — operator-facing notifications (would fan out to Amazon SNS on AWS)
ALERTS: deque = deque(maxlen=50)
ALERT_LOCK = threading.Lock()
_alert_id = 0


def alert(level: str, msg: str) -> None:
    """level: critical | warning | info | ok. Skips an immediate duplicate."""
    global _alert_id
    with ALERT_LOCK:
        if ALERTS and ALERTS[0]["msg"] == msg:
            return
        _alert_id += 1
        ALERTS.appendleft({"id": _alert_id, "t": time.strftime("%H:%M:%S"), "level": level, "msg": msg})


# --------------------------------------------------------------------------- #
# Incidents (global, across both feeds)
# --------------------------------------------------------------------------- #
@dataclass
class Incident:
    id: int
    feed: str
    feed_name: str
    ts: float
    confidence: float
    severity: str                 # advisory | warning | critical
    status: str = "awaiting_approval"


INCIDENTS: list[Incident] = []
INC_LOCK = threading.Lock()
_inc_id = 0


def new_incident(feed, name, conf, severity) -> Incident:
    global _inc_id
    with INC_LOCK:
        _inc_id += 1
        inc = Incident(_inc_id, feed, name, time.time(), conf, severity)
        INCIDENTS.insert(0, inc)
        del INCIDENTS[40:]
    return inc


# --------------------------------------------------------------------------- #
# A feed worker
# --------------------------------------------------------------------------- #
@dataclass
class Feed:
    id: str
    name: str
    role: str
    video: str
    kind: str                     # "beach" | "sea"
    zone_frac: list | None = None  # beach only; defaults to ZONE_FRAC
    lock: threading.Lock = field(default_factory=threading.Lock)
    jpeg: bytes = b""
    mask_jpeg: bytes = b""
    frame_no: int = 0
    fps: float = 0.0
    proc_ms: float = 0.0
    stages: dict = field(default_factory=dict)
    trace: deque = field(default_factory=lambda: deque(maxlen=40))
    pipeline: deque = field(default_factory=lambda: deque(maxlen=40))
    tool_calls: dict = field(default_factory=lambda: {
        "VideoCapture.read": 0, "cvtColor": 0, "GaussianBlur": 0,
        "MOG2.apply": 0, "morphologyEx": 0, "findContours": 0,
        "pointPolygonTest": 0, "Canny": 0, "absdiff": 0, "threshold": 0, "imencode": 0,
    })
    # live readings
    people: int = 0
    in_zone: int = 0
    last_conf: float = 0.0
    # sea feed
    sea_state: str = "calm"
    whitewater: float = 0.0
    rip_score: int = 0

    def log(self, dq: str, step: str, detail: str, tool: str = "") -> None:
        tgt = self.trace if dq == "trace" else self.pipeline
        with self.lock:
            tgt.appendleft({"t": time.strftime("%H:%M:%S"), "step": step, "tool": tool, "detail": detail})

    def bump(self, **kw) -> None:
        with self.lock:
            for k, v in kw.items():
                self.tool_calls[k] = self.tool_calls.get(k, 0) + v


FEEDS: dict[str, Feed] = {}


def encode(img, q=72):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return buf.tobytes() if ok else b""


# ---- Beach feed pipeline --------------------------------------------------- #
def run_beach(feed: Feed, frame, bg, kernel, zone):
    rec, stages = stage_timer()
    h, w = frame.shape[:2]
    rec("grayscale · cv2.cvtColor", lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    blur = rec("denoise · cv2.GaussianBlur", lambda: cv2.GaussianBlur(frame, (5, 5), 0))
    fg = rec("bg model · cv2.MOG2.apply", lambda: bg.apply(blur))
    fg = rec("shadow gate · cv2.threshold", lambda: cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)[1])
    fg = rec("morphology · cv2.morphologyEx", lambda: cv2.morphologyEx(
        cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel), cv2.MORPH_DILATE, kernel, iterations=2))
    cnts = rec("contours · cv2.findContours", lambda: cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    feed.bump(**{"cvtColor": 1, "GaussianBlur": 1, "MOG2.apply": 1, "threshold": 1,
                 "morphologyEx": 2, "findContours": 1})

    def filt():
        out = []
        for c in cnts:
            a = cv2.contourArea(c) / (w * h)
            if a >= MIN_AREA_FRAC:
                x, y, bw, bh = cv2.boundingRect(c)
                out.append((x, y, bw, bh, a))
        return out
    boxes = rec("blob filter · cv2.contourArea", filt)

    def geofence():
        ins = []
        for b in boxes:
            x, y, bw, bh, _ = b
            feed.bump(pointPolygonTest=1)
            if cv2.pointPolygonTest(zone, (float(x + bw / 2), float(y + bh)), False) >= 0:
                ins.append(b)
        return ins
    intr = rec("geofence · cv2.pointPolygonTest", geofence)
    intruder = max(intr, key=lambda b: b[4]) if intr else None

    info = {"confidence": 0.0, "aspect": 0.0}
    if intruder is not None:
        def inspect():
            x, y, bw, bh, a = intruder
            roi = frame[max(0, y):y + bh, max(0, x):x + bw]
            if roi.size == 0:
                return {"confidence": 0.0, "aspect": 0.0}
            g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            feed.bump(Canny=1, cvtColor=1)
            ed = float(cv2.Canny(g, 60, 160).mean()) / 255.0
            asp = bh / max(bw, 1)
            up = 1.0 if 1.1 <= asp <= 4.0 else 0.4
            return {"confidence": round(min(1.0, a / CONFIRM_FRAC) * (0.5 + ed) * up, 2), "aspect": round(asp, 2)}
        info = rec("ROI inspect · cv2.Canny", inspect)

    with feed.lock:
        feed.people = len(boxes)
        feed.in_zone = len(intr)
        feed.last_conf = info["confidence"]
        feed.stages = stages
    return boxes, intruder, info, fg


def draw_beach(frame, zone, boxes, intruder, info, critical):
    cv2.polylines(frame, [zone], True, (0, 160, 255), 2)
    cv2.putText(frame, "RESTRICTED WATER ZONE", (12, int(zone[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 160, 255), 2)
    for b in boxes:
        x, y, bw, bh, _ = b
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (70, 200, 70), 1)
    if intruder is not None:
        x, y, bw, bh, _ = intruder
        hot = info["confidence"] >= CONF_GATE
        col = (0, 60, 255) if (hot and critical) else (0, 165, 255) if hot else (0, 210, 255)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), col, 3)
        lbl = ("CRITICAL · SWIMMER IN ZONE" if critical else "SWIMMER IN ZONE") if hot else "inspecting…"
        cv2.putText(frame, lbl, (x, max(18, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return frame


# ---- Sea-state feed pipeline ---------------------------------------------- #
def run_sea(feed: Feed, frame, bg, prev_gray, ema):
    rec, stages = stage_timer()
    h, w = frame.shape[:2]
    gray = rec("grayscale · cv2.cvtColor", lambda: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    diff = rec("frame-diff · cv2.absdiff", lambda: cv2.absdiff(gray, prev_gray) if prev_gray is not None else gray * 0)
    motion = rec("motion magnitude · np.mean", lambda: float(diff.mean()))
    foam = rec("whitewater · cv2.threshold", lambda: cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1])
    lower = foam[int(h * 0.45):, :]
    cover = rec("coverage · np.mean", lambda: float(lower.mean()) / 255.0)
    fg = rec("bg model · cv2.MOG2.apply", lambda: bg.apply(frame))
    feed.bump(**{"cvtColor": 1, "absdiff": 1, "threshold": 1, "MOG2.apply": 1})

    # surf index fuses wave motion (frame-diff) and whitewater coverage,
    # normalised to this camera's range, then smoothed.
    mnorm = max(0.0, min(1.0, motion / 2.0))
    cnorm = max(0.0, min(1.0, (cover - 0.05) / 0.05))
    raw = 100.0 * (0.5 * mnorm + 0.5 * cnorm)
    e = ema[0] = 0.8 * ema[0] + 0.2 * raw
    rip = rec("surf index / rip model", lambda: int(min(100, e)))
    if rip > 62:
        sea, risk, lvl = "rough", "high", "HIGH"
    elif rip > 35:
        sea, risk, lvl = "moderate", "moderate", "ELEVATED"
    else:
        sea, risk, lvl = "calm", "low", "CALM"

    def publish():
        prev = POSTURE.level
        POSTURE.level, POSTURE.sea_state, POSTURE.rip_risk = lvl, sea, risk
        POSTURE.rip_score, POSTURE.whitewater = rip, round(cover, 3)
        if lvl != prev:
            fusion(f"Sea-state cam → {sea.upper()} sea, rip risk {risk.upper()} "
                   f"(whitewater {cover*100:.0f}%). Beach risk posture: {prev} → {lvl}.")
            if lvl == "HIGH":
                alert("warning", f"Rip risk HIGH at {feed.name} — advise red flag, no swimming.")
        return lvl
    rec("publish posture (fusion)", publish)

    with feed.lock:
        feed.sea_state, feed.whitewater, feed.rip_score = sea, round(cover, 3), rip
        feed.stages = stages
    return gray, fg, sea, rip, cover


def draw_sea(frame, sea, rip, cover):
    h = frame.shape[0]
    col = (0, 60, 255) if sea == "rough" else (0, 165, 255) if sea == "moderate" else (80, 200, 120)
    cv2.rectangle(frame, (0, h - 54), (430, h), (16, 16, 20), -1)
    cv2.putText(frame, f"SEA STATE: {sea.upper()}", (12, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    cv2.putText(frame, f"rip risk {rip}/100  ·  whitewater {cover*100:.0f}%", (12, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 220), 1)
    return frame


def mask_preview(fg, label):
    m = cv2.cvtColor(fg, cv2.COLOR_GRAY2BGR)
    m[np.where((fg > 0))] = (255, 190, 60)   # cyan-ish foreground
    cv2.rectangle(m, (0, 0), (m.shape[1], 24), (10, 10, 12), -1)
    cv2.putText(m, label, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 220), 1)
    return m


# --------------------------------------------------------------------------- #
# Worker loops
# --------------------------------------------------------------------------- #
def beach_loop(feed: Feed):
    bg = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=28, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    zfrac = feed.zone_frac or ZONE_FRAC
    zone = None
    open_inc: Incident | None = None
    cooldown = 0
    last_t = time.perf_counter()
    heartbeat = 0
    while True:
        cap = cv2.VideoCapture(feed.video)
        while True:
            ok, frame = cap.read()
            feed.bump(**{"VideoCapture.read": 1})
            if not ok:
                break
            if zone is None:
                h, w = frame.shape[:2]
                zone = np.array([(int(x * w), int(y * h)) for x, y in zfrac], np.int32)
            t0 = time.perf_counter()
            feed.frame_no += 1
            boxes, intruder, info, fg = run_beach(feed, frame, bg, kernel, zone)

            # resolve open incident
            if open_inc is not None and open_inc.status != "awaiting_approval":
                verb = "APPROVED — lifeguard dispatched" if open_inc.status == "dispatched" else "dismissed — false alarm"
                feed.log("trace", "human", f"#{open_inc.id} operator {verb}")
                if open_inc.status == "dispatched":
                    alert("ok", f"Lifeguard dispatched to {feed.name} (incident #{open_inc.id}).")
                open_inc = None
                cooldown = int(TARGET_FPS * 5)

            critical = POSTURE.level == "HIGH"
            if cooldown > 0:
                cooldown -= 1
            elif intruder is not None and open_inc is None:
                feed.log("trace", "perceive",
                         f"motion in restricted zone (area {intruder[4]*100:.2f}% of frame)", "detect_motion+geofence")
                feed.log("trace", "decide",
                         f"re-measured ROI: confidence {info['confidence']}, aspect {info['aspect']}", "inspect/Canny")
                if info["confidence"] >= CONF_GATE:
                    sev = "critical" if critical else "warning"
                    if critical:
                        feed.log("trace", "fuse",
                                 f"cross-feed: sea-state cam reports {POSTURE.sea_state.upper()} "
                                 f"(rip {POSTURE.rip_risk.upper()}) → severity {sev.upper()}", "posture")
                    open_inc = new_incident(feed.id, feed.name, info["confidence"], sev)
                    feed.log("trace", "act",
                             f"#{open_inc.id} swimmer in zone — {sev.upper()} — REQUESTING OPERATOR APPROVAL", "escalate")
                    fusion(f"Beach cam: swimmer confirmed in restricted zone (conf {info['confidence']}), "
                           f"severity {sev.upper()} under {POSTURE.level} posture → operator approval requested.")
                    alert("critical" if sev == "critical" else "warning",
                          f"{sev.upper()}: swimmer in restricted zone at {feed.name} (incident #{open_inc.id}, conf {info['confidence']}).")
                else:
                    feed.log("trace", "decide", "below confidence gate — no action (surf glint)", "inspect")

            shown = draw_beach(frame.copy(), zone, boxes, intruder, info, critical)
            banner(shown)
            with feed.lock:
                feed.jpeg = encode(shown)
                feed.mask_jpeg = encode(mask_preview(fg, "MOG2 foreground · OpenCV 5"), 60)
            feed.bump(imencode=2)

            heartbeat += 1
            if heartbeat % 6 == 0:
                feed.log("pipeline", "cv",
                         f"f{feed.frame_no} · MOG2→{feed.people} blobs · geofence→{feed.in_zone} in zone "
                         f"· {sum(feed.stages.values()):.1f}ms", "OpenCV5")

            now = time.perf_counter()
            feed.fps = 0.85 * feed.fps + 0.15 * (1.0 / max(now - last_t, 1e-3))
            feed.proc_ms = 0.85 * feed.proc_ms + 0.15 * ((now - t0) * 1000)
            last_t = now
            time.sleep(1.0 / TARGET_FPS)
        cap.release()


def sea_loop(feed: Feed):
    bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24, detectShadows=False)
    prev_gray = None
    ema = [8.0]
    last_t = time.perf_counter()
    heartbeat = 0
    while True:
        cap = cv2.VideoCapture(feed.video)
        while True:
            ok, frame = cap.read()
            feed.bump(**{"VideoCapture.read": 1})
            if not ok:
                break
            t0 = time.perf_counter()
            feed.frame_no += 1
            gray, fg, sea, rip, cover = run_sea(feed, frame, bg, prev_gray, ema)
            prev_gray = gray
            shown = draw_sea(frame.copy(), sea, rip, cover)
            banner(shown)
            with feed.lock:
                feed.jpeg = encode(shown)
                feed.mask_jpeg = encode(mask_preview(fg, "surf motion · OpenCV 5"), 60)
            feed.bump(imencode=2)

            heartbeat += 1
            if heartbeat % 6 == 0:
                feed.log("pipeline", "cv",
                         f"f{feed.frame_no} · Δ-motion → surf energy {ema[0]:.1f} · whitewater {cover*100:.0f}% "
                         f"· sea {sea.upper()}", "OpenCV5")
            if heartbeat % 24 == 0:
                feed.log("trace", "monitor", f"sea state {sea.upper()}, rip risk {rip}/100, whitewater {cover*100:.0f}%", "sea-model")

            now = time.perf_counter()
            feed.fps = 0.85 * feed.fps + 0.15 * (1.0 / max(now - last_t, 1e-3))
            feed.proc_ms = 0.85 * feed.proc_ms + 0.15 * ((now - t0) * 1000)
            last_t = now
            time.sleep(1.0 / TARGET_FPS)
        cap.release()


def banner(frame):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (18, 18, 22), -1)
    cv2.putText(frame, f"Neonet Vision Agent  ·  OpenCV {cv2.__version__}  ·  agentic perception→decision→action",
                (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 220), 1)


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="Neonet Vision Agent")


DEFAULT_FEEDS = [
    {"id": "beach", "name": "Rushikonda Cove — Beach Cam", "role": "zone intrusion",
     "video": "media/demo.mp4", "kind": "beach", "zone_frac": ZONE_FRAC},
    {"id": "sea", "name": "Offshore — Sea-State Cam", "role": "sea state / rip risk",
     "video": "media/demo2.mp4", "kind": "sea"},
]


def load_feed_config() -> list[dict]:
    """Feeds come from feeds.json if present, else the two bundled defaults.
    Each entry: {id, name, role, video, kind: 'beach'|'sea', zone_frac?}.
    `video` is a path relative to the repo (or absolute). Drop an .mp4 in
    media/, add an entry here, restart — see TESTING in the README."""
    cfg = HERE / "feeds.json"
    if cfg.exists():
        try:
            data = json.loads(cfg.read_text())
            if isinstance(data, list) and data:
                return data
            print("[config] feeds.json is empty/invalid — using defaults")
        except Exception as e:
            print(f"[config] feeds.json parse error ({e}) — using defaults")
    return DEFAULT_FEEDS


@app.on_event("startup")
def _start():
    for fc in load_feed_config():
        vid = fc["video"]
        vid = vid if Path(vid).is_absolute() else str(HERE / vid)
        kind = fc.get("kind", "beach")
        feed = Feed(fc["id"], fc["name"], fc.get("role", kind), vid, kind, fc.get("zone_frac"))
        if not Path(vid).exists():
            print(f"[config] feed '{feed.id}': video not found: {vid} — skipping")
            continue
        FEEDS[feed.id] = feed
        loop = beach_loop if kind == "beach" else sea_loop
        threading.Thread(target=loop, args=(feed,), daemon=True).start()
        print(f"[config] started feed '{feed.id}' ({kind}) <- {vid}")
    if not FEEDS:
        print("[config] no valid feeds — check feeds.json / media paths")


def mjpeg(getter):
    b = b"--frame"
    while True:
        buf = getter()
        if buf:
            yield b + b"\r\nContent-Type: image/jpeg\r\n\r\n" + buf + b"\r\n"
        time.sleep(1.0 / TARGET_FPS)


@app.get("/stream/{fid}")
def stream(fid: str):
    f = FEEDS.get(fid)
    return StreamingResponse(mjpeg(lambda: f.jpeg), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/mask/{fid}")
def mask(fid: str):
    f = FEEDS.get(fid)
    return StreamingResponse(mjpeg(lambda: f.mask_jpeg), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/state")
def state():
    feeds = {}
    for fid, f in FEEDS.items():
        with f.lock:
            feeds[fid] = {
                "id": fid, "name": f.name, "role": f.role, "kind": f.kind,
                "frame": f.frame_no, "fps": round(f.fps, 1), "proc_ms": round(f.proc_ms, 1),
                "stages": f.stages, "tool_calls": dict(f.tool_calls),
                "trace": list(f.trace)[:14], "pipeline": list(f.pipeline)[:12],
                "people": f.people, "in_zone": f.in_zone, "last_conf": f.last_conf,
                "sea_state": f.sea_state, "whitewater": f.whitewater, "rip_score": f.rip_score,
            }
    with INC_LOCK:
        incs = [vars(i) for i in INCIDENTS]
    with FUSION_LOCK:
        flog = list(FUSION_LOG)
    with ALERT_LOCK:
        alerts = list(ALERTS)[:14]
    in_water = FEEDS["beach"].in_zone if "beach" in FEEDS else 0
    base = {"CALM": 20, "ELEVATED": 55, "HIGH": 80}[POSTURE.level]
    beach_risk = min(100, base + in_water * 8)
    total_tools = sum(sum(f.tool_calls.values()) for f in FEEDS.values())
    return JSONResponse({
        "cv": cv2.__version__,
        "posture": vars(POSTURE),
        "beach_risk": beach_risk,
        "total_tool_calls": total_tools,
        "feeds": feeds,
        "incidents": incs,
        "fusion": flog,
        "alerts": alerts,
        "counts": {
            "active": sum(1 for i in incs if i["status"] == "awaiting_approval"),
            "dispatched": sum(1 for i in incs if i["status"] == "dispatched"),
            "dismissed": sum(1 for i in incs if i["status"] == "dismissed"),
        },
    })


@app.post("/incident/{iid}/{decision}")
def decide(iid: int, decision: str):
    with INC_LOCK:
        for inc in INCIDENTS:
            if inc.id == iid and inc.status == "awaiting_approval":
                inc.status = "dispatched" if decision == "approve" else "dismissed"
    return JSONResponse({"ok": True})


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


HTML = (HERE / "ui.html").read_text()
