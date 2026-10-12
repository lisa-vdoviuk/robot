#!/usr/bin/env python3
"""
Rescue Robot dashboard server.
Run from the project root: python3 backend/server.py -> https://<pi-ip>:8000
Keep settings.json and model files in the project root.
HTTPS certificates may be in the project root, backend/ or the launch directory.
Install: sudo apt install python3-flask python3-picamera2 python3-gpiozero python3-lgpio alsa-utils
         pip install ultralytics ncnn --break-system-packages
"""
import array, atexit, io, json, math, os, queue, re, ssl, struct, subprocess, tempfile, threading, time, wave
from flask import Flask, Response, jsonify, request, send_file
from gpiozero import DistanceSensor, Motor

# ---------- CONFIG ----------
LEFT = dict(forward=16, backward=26, enable=13)
RIGHT = dict(forward=5, backward=6, enable=12)
SONARS = [dict(echo=24, trigger=23), dict(echo=27, trigger=17)]

# User confirmed: original S2 faces forward; original S1 faces backward.
FRONT_SONAR, REAR_SONAR = 1, 0

SPK = "plughw:CARD=MAX98357A,DEV=0"
MIC = "auto"
CLASSES = {0: "person", 16: "dog"}
COLORS = {"person": (176, 198, 60), "dog": (36, 165, 245)}
MODEL = "yolo11n"

# Resolve shared files independently of the working directory.
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_FILE = os.path.join(HERE, "dashboard", "dashboard.html")
CFG_FILE = os.path.join(HERE, "settings.json")

DEFAULTS = dict(
    stop_cm=20,
    watchdog=0.4,
    max_speed=1.0,
    trim_l=1.0,
    trim_r=1.0,
    swap=False,
    flip_camera=True,
    det_conf=0.4,
    confirm=3,
    imgsz=320,
    talk_gain=1.0,
)

CFG = dict(DEFAULTS)
try:
    CFG.update({
        k: v
        for k, v in json.load(open(CFG_FILE)).items()
        if k in DEFAULTS
    })
except Exception:
    pass

app = Flask(__name__)

S = dict(
    l=0.0,
    r=0.0,
    t=0.0,
    safety=True,
    blocked=False,
    testing=False,
    dist=[None, None],
    fps=0.0,
    cam=False,
    detect=False,
    det_state="off",
    det_fps=0.0,
    infer_ms=0,
    alert=[],
    talk_t=0.0,
    blocked_by=[],
    applied_drive=[0.0, 0.0],
)

frame, out = dict(jpg=None), dict(jpg=None)
cond = threading.Condition()
talk_q = queue.Queue(maxsize=60)


def mk(cls, **kw):
    try:
        return cls(**kw)
    except Exception as e:
        print("Hardware unavailable:", kw, e)
        return None


motors = [
    mk(Motor, pwm=True, **LEFT),
    mk(Motor, pwm=True, **RIGHT),
]

sonars = [
    mk(DistanceSensor, max_distance=3, **p)
    for p in SONARS
]


def drive(m, v):
    if m is None:
        return

    v = max(-1.0, min(1.0, v))

    if v > 0:
        m.forward(v)
    elif v < 0:
        m.backward(-v)
    else:
        m.stop()


def sonar_ranges():
    return S["dist"][FRONT_SONAR], S["dist"][REAR_SONAR]


def obstacle_blocks(l, r, front_cm, rear_cm, stop_cm, enabled):
    """Block the whole command if either wheel moves toward a near obstacle.

    Pivot turns involve both directions, so either near sensor blocks them.
    These two sensors do not measure side clearance.
    """
    if not enabled:
        return []

    near = lambda d: (
        d is not None
        and math.isfinite(d)
        and 0 <= d <= stop_cm
    )

    blocked = []

    if (l > 0 or r > 0) and near(front_cm):
        blocked.append("front")

    if (l < 0 or r < 0) and near(rear_cm):
        blocked.append("rear")

    return blocked


def control():
    while True:
        l, r = S["l"], S["r"]

        if time.time() - S["t"] > CFG["watchdog"]:
            l = r = 0

        if not S["testing"]:
            m = CFG["max_speed"]
            a, b = (r, l) if CFG["swap"] else (l, r)

            a = max(
                -1.0,
                min(1.0, max(-m, min(m, a)) * CFG["trim_l"]),
            )
            b = max(
                -1.0,
                min(1.0, max(-m, min(m, b)) * CFG["trim_r"]),
            )

            front_cm, rear_cm = sonar_ranges()

            S["blocked_by"] = obstacle_blocks(
                a,
                b,
                front_cm,
                rear_cm,
                CFG["stop_cm"],
                S["safety"],
            )
            S["blocked"] = bool(S["blocked_by"])

            if S["blocked"]:
                a = b = 0.0

            S["applied_drive"] = [a, b]

            drive(motors[0], a)
            drive(motors[1], b)

        time.sleep(0.05)


def sonar_loop():
    hist = [[], []]

    while True:
        for i, s in enumerate(sonars):
            if s is not None:
                hist[i] = (hist[i] + [s.distance * 100])[-3:]
                S["dist"][i] = round(
                    sorted(hist[i])[len(hist[i]) // 2],
                    1,
                )
                time.sleep(0.05)

        time.sleep(0.05)


def camera_loop():
    try:
        from picamera2 import Picamera2
        from libcamera import Transform

        cam = Picamera2()
        cam.configure(
            cam.create_video_configuration(
                main={"size": (640, 480)},
                transform=Transform(
                    hflip=int(CFG["flip_camera"]),
                    vflip=int(CFG["flip_camera"]),
                ),
            )
        )
        cam.options["quality"] = 70
        cam.start()
    except Exception as e:
        print("Camera unavailable:", e)
        return

    S["cam"] = True
    n, t0 = 0, time.time()

    while True:
        buf = io.BytesIO()
        cam.capture_file(buf, format="jpeg")

        with cond:
            frame["jpg"] = buf.getvalue()
            if not S["detect"]:
                out["jpg"] = frame["jpg"]
            cond.notify_all()

        n += 1
        if time.time() - t0 >= 2:
            S["fps"] = round(n / (time.time() - t0), 1)
            n, t0 = 0, time.time()


def load_model():
    from ultralytics import YOLO

    ncnn = f"{MODEL}_ncnn_model"

    try:
        if not os.path.isdir(os.path.join(HERE, ncnn)):
            YOLO(os.path.join(HERE, f"{MODEL}.pt")).export(
                format="ncnn",
                imgsz=320,
            )

        return YOLO(os.path.join(HERE, ncnn), task="detect")
    except Exception as e:
        print("NCNN unavailable, using PyTorch:", e)
        return YOLO(os.path.join(HERE, f"{MODEL}.pt"))


def detect_loop():
    model = None
    streak = {c: 0 for c in CLASSES.values()}
    n, t0 = 0, time.time()

    while True:
        if not S["detect"] or frame["jpg"] is None:
            S["alert"], n, t0 = [], 0, time.time()
            time.sleep(0.2)
            continue

        try:
            import cv2
            import numpy as np

            if model is None:
                S["det_state"] = "loading model..."
                model = load_model()

            S["det_state"] = "running"

            img = cv2.imdecode(
                np.frombuffer(frame["jpg"], np.uint8),
                cv2.IMREAD_COLOR,
            )

            t = time.time()
            res = model(
                img,
                classes=list(CLASSES),
                conf=CFG["det_conf"],
                imgsz=CFG["imgsz"],
                verbose=False,
            )[0]

            S["infer_ms"] = round((time.time() - t) * 1000)
            seen = set()

            for b in res.boxes:
                name = CLASSES[int(b.cls)]
                seen.add(name)

                x1, y1, x2, y2 = map(int, b.xyxy[0])
                col = COLORS[name]
                txt = f"{name.capitalize()} {float(b.conf):.0%}"

                cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)

                (tw, th), _ = cv2.getTextSize(
                    txt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2,
                )

                ty = y1 if y1 > th + 8 else y1 + th + 8

                cv2.rectangle(
                    img,
                    (x1, ty - th - 8),
                    (x1 + tw + 8, ty),
                    col,
                    -1,
                )

                cv2.putText(
                    img,
                    txt,
                    (x1 + 4, ty - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (20, 20, 20),
                    2,
                )

            for c in streak:
                streak[c] = streak[c] + 1 if c in seen else 0

            S["alert"] = [
                c for c, k in streak.items()
                if k >= CFG["confirm"]
            ]

            with cond:
                out["jpg"] = cv2.imencode(
                    ".jpg",
                    img,
                    [cv2.IMWRITE_JPEG_QUALITY, 70],
                )[1].tobytes()
                cond.notify_all()

            n += 1
            if time.time() - t0 >= 2:
                S["det_fps"] = round(n / (time.time() - t0), 1)
                n, t0 = 0, time.time()

        except Exception as e:
            S["det_state"] = f"error: {str(e)[:90]}"
            S["detect"] = False
            model = None
            print("Detection error:", e)


def sh(cmd):
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3,
        ).stdout.strip()
    except Exception:
        return ""


def sysinfo():
    m = {
        l.split(":")[0]: int(l.split()[1])
        for l in open("/proc/meminfo")
        if ":" in l
    }

    return (
        round(100 * (1 - m["MemAvailable"] / m["MemTotal"])),
        round(os.getloadavg()[0] / os.cpu_count() * 100),
    )


# ---------- TESTS ----------
def beep(path, freq=440, dur=1.2, rate=48000):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)

        w.writeframes(
            b"".join(
                struct.pack("<hh", v, v)
                for v in (
                    int(12000 * math.sin(2 * math.pi * freq * i / rate))
                    for i in range(int(rate * dur))
                )
            )
        )


def play(path):
    return subprocess.run(
        ["aplay", "-q", "-D", SPK, path],
        capture_output=True,
        text=True,
        timeout=15,
    )


def find_mic():
    if MIC != "auto":
        return MIC

    cards = re.findall(
        r"card (\d+): (\S+) \[(.*?)\], device (\d+):",
        sh(["arecord", "-l"]),
    )

    if not cards:
        return None

    pick = next(
        (
            c for c in cards
            if "usb" in (c[1] + c[2]).lower()
        ),
        cards[0],
    )

    return f"plughw:CARD={pick[1]},DEV={pick[3]}"


def record(sec):
    path = os.path.join(tempfile.gettempdir(), "mic.wav")
    dev = find_mic()

    if dev is None:
        return path, subprocess.CompletedProcess(
            [],
            1,
            "",
            "no microphone found (arecord -l is empty): plug in the USB mic",
        )

    r = subprocess.run(
        [
            "arecord", "-q", "-D", dev,
            "-d", str(sec),
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            path,
        ],
        capture_output=True,
        text=True,
        timeout=sec + 5,
    )

    return path, r


def levels(path):
    with wave.open(path) as w:
        a = array.array("h", w.readframes(w.getnframes()))

    db = lambda v: 20 * math.log10(max(v, 1) / 32768)

    return (
        db(math.sqrt(sum(x * x for x in a) / max(len(a), 1))),
        db(max((abs(x) for x in a), default=0)),
    )


def t_sonar():
    d = sonar_ranges()
    ok = all(x is not None and 2 < x < 299 for x in d)

    return (
        ok,
        f"Front {d[0]} cm, rear {d[1]} cm"
        + ("" if ok else " (no echo: check Echo wire, divider, 5V)"),
    )


def t_camera():
    j = frame["jpg"]

    return (
        bool(j),
        f"frame {len(j)//1024} KB, {S['fps']} fps"
        if j else "no frames (check ribbon cable / picamera2)",
    )


def t_speaker():
    p = os.path.join(tempfile.gettempdir(), "beep.wav")
    beep(p)
    r = play(p)

    return (
        r.returncode == 0,
        "440 Hz tone played, did you hear it?"
        if r.returncode == 0 else r.stderr.strip()[:120],
    )


def t_mic():
    p, r = record(2)

    if r.returncode != 0:
        return False, r.stderr.strip()[:120] or "arecord failed"

    rms, peak = levels(p)

    return (
        peak > -40,
        f"level {rms:.0f} dB, peak {peak:.0f} dB"
        + (
            ""
            if peak > -40
            else " (too quiet: speak or clap while recording)"
        ),
    )


def t_echo():
    p, r = record(3)

    if r.returncode != 0:
        return False, "recording failed"

    return play(p).returncode == 0, "recorded 3 s and played it back"


def t_motors():
    for i in (0, 1):
        for v in (0.4, -0.4):
            drive(motors[i], v)
            time.sleep(0.5)
            drive(motors[i], 0)
            time.sleep(0.3)

    return (
        all(m is not None for m in motors),
        "left fwd/back, then right fwd/back (watch the wheels)",
    )


TESTS = dict(
    sonar=t_sonar,
    camera=t_camera,
    speaker=t_speaker,
    mic=t_mic,
    echo=t_echo,
    motors=t_motors,
)


# ---------- ROUTES ----------
@app.route("/")
def index():
    return send_file(DASHBOARD_FILE)


@app.route("/video")
def video():
    def gen():
        while True:
            with cond:
                cond.wait(timeout=1)
                j = out["jpg"]

            if j:
                yield (
                    b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                    + j
                    + b"\r\n"
                )

    return Response(
        gen(),
        mimetype="multipart/x-mixed-replace; boundary=f",
    )


@app.post("/api/drive")
def api_drive():
    j = request.get_json(force=True)
    S.update(
        l=float(j.get("l", 0)),
        r=float(j.get("r", 0)),
        t=time.time(),
    )
    return jsonify(ok=True)


@app.post("/api/stop")
def api_stop():
    S.update(l=0.0, r=0.0, t=0.0)
    return jsonify(ok=True)


@app.post("/api/safety")
def api_safety():
    S["safety"] = bool(request.get_json(force=True).get("on", True))
    return jsonify(ok=True)


@app.post("/api/detect")
def api_detect():
    S["detect"] = bool(request.get_json(force=True).get("on", False))
    S["det_state"] = "starting..." if S["detect"] else "off"
    return jsonify(ok=True)


def talk_worker():
    """Play operator PCM in a dedicated worker thread."""
    p = None

    while True:
        chunk = talk_q.get()

        if chunk is None:
            if p:
                try:
                    p.stdin.close()
                    p.wait(timeout=3)
                except Exception:
                    p.kill()

            p = None
            continue

        if p is None or p.poll() is not None:
            p = subprocess.Popen(
                [
                    "aplay", "-q", "-D", SPK,
                    "-t", "raw",
                    "-f", "S16_LE",
                    "-r", "16000",
                    "-c", "1",
                    "-B", "200000",
                ],
                stdin=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            S["talk_err"] = ""
            print("talk: aplay started")

        try:
            p.stdin.write(chunk)
            p.stdin.flush()
        except Exception as e:
            err = (
                p.stderr.read().decode(errors="ignore")[:160]
                if p.poll() is not None else ""
            )

            S["talk_err"] = (err or str(e)).strip()
            print("talk: speaker error:", S["talk_err"])
            p = None


@app.post("/api/talk")
def api_talk():
    try:
        talk_q.put_nowait(request.get_data())
    except queue.Full:
        pass

    S["talk_t"] = time.time()
    return "", 204


@app.post("/api/talk/stop")
def api_talk_stop():
    try:
        talk_q.put(None, timeout=1)
    except queue.Full:
        pass

    S["talk_t"] = 0
    return "", 204


@app.get("/api/status")
def api_status():
    mem, cpu = sysinfo()
    t = (
        sh(["vcgencmd", "measure_temp"])
        .replace("temp=", "")
        .replace("'C", "")
    )
    front_cm, rear_cm = sonar_ranges()

    return jsonify(
        dist=S["dist"],
        front_cm=front_cm,
        rear_cm=rear_cm,
        blocked_by=S["blocked_by"],
        applied_drive=S["applied_drive"],
        blocked=S["blocked"],
        safety=S["safety"],
        fps=S["fps"],
        cam=S["cam"],
        temp=t,
        throttled=sh(["vcgencmd", "get_throttled"]).replace(
            "throttled=", ""
        ),
        cpu=cpu,
        mem=mem,
        drive=[S["l"], S["r"]],
        testing=S["testing"],
        stop_cm=CFG["stop_cm"],
        detect=S["detect"],
        det_state=S["det_state"],
        det_fps=S["det_fps"],
        infer_ms=S["infer_ms"],
        alert=S["alert"],
        talking=time.time() - S["talk_t"] < 1,
        talk_err=S.get("talk_err", ""),
    )


@app.post("/api/test/<name>")
def api_test(name):
    fn = TESTS.get(name)

    if not fn:
        return jsonify(ok=False, msg="unknown test"), 404

    if S["testing"]:
        return jsonify(ok=False, msg="another test is still running")

    S["testing"] = True

    for m in motors:
        drive(m, 0)

    try:
        ok, msg = fn()
    except Exception as e:
        ok, msg = False, str(e)[:160]
    finally:
        S["testing"] = False

    return jsonify(ok=bool(ok), msg=msg)


@app.get("/api/settings")
def api_settings_get():
    return jsonify(cfg=CFG, defaults=DEFAULTS)


@app.post("/api/settings")
def api_settings_set():
    j = request.get_json(force=True)

    if j.get("reset"):
        CFG.update(DEFAULTS)

    for k, v in j.items():
        if k in DEFAULTS:
            CFG[k] = type(DEFAULTS[k])(v)

    json.dump(CFG, open(CFG_FILE, "w"), indent=2)
    return jsonify(ok=True, cfg=CFG)


def cleanup():
    for m in motors:
        if m:
            m.stop()


def https_context():
    # Reuse certificates after moving server.py into backend/.
    directories = dict.fromkeys((
        HERE,
        os.path.dirname(os.path.abspath(__file__)),
        os.getcwd(),
    ))

    for directory in directories:
        cert = os.path.join(directory, "cert.pem")
        key = os.path.join(directory, "key.pem")

        if os.path.isfile(cert) and os.path.isfile(key):
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(cert, key)

            print(
                f"HTTPS enabled; certificates: {directory}",
                flush=True,
            )
            return context

    raise RuntimeError(
        "HTTPS certificates not found. Keep cert.pem and key.pem "
        "together in the project root or backend/. "
        "To create them, run from the project root: "
        "openssl req -x509 -newkey rsa:2048 -nodes "
        "-keyout key.pem -out cert.pem "
        "-days 365 -subj /CN=robot.local"
    )


if __name__ == "__main__":
    tls = https_context()
    atexit.register(cleanup)

    for f in (
        control,
        sonar_loop,
        camera_loop,
        detect_loop,
        talk_worker,
    ):
        threading.Thread(target=f, daemon=True).start()

    app.run(
        host="0.0.0.0",
        port=8000,
        threaded=True,
        ssl_context=tls,
    )