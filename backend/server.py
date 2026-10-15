#!/usr/bin/env python3
"""
Run from the project root:
    python3 backend/server.py

Open:
    https://<pi-ip>:8000
"""
import array
import atexit
import io
import json
import math
import os
import queue
import re
import signal
import ssl
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave

from flask import Flask, Response, jsonify, request, send_file
from gpiozero import DistanceSensor, Motor
from imu import MPU6050


# ---------- CONFIG ----------

LEFT = dict(forward=16, backward=26, enable=13)
RIGHT = dict(forward=5, backward=6, enable=12)

SONARS = [
    dict(echo=24, trigger=23),
    dict(echo=27, trigger=17),
]
FRONT_SONAR, REAR_SONAR = 1, 0

SPK = "plughw:CARD=MAX98357A,DEV=0"
MIC = "auto"

CLASSES = {0: "person", 16: "dog"}
COLORS = {
    "person": (176, 198, 60),
    "dog": (36, 165, 245),
}
MODEL = "yolo11n"

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
    with open(CFG_FILE) as f:
        CFG.update({
            k: v for k, v in json.load(f).items()
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

frame = dict(jpg=None)
out = dict(jpg=None)
cond = threading.Condition()
talk_q = queue.Queue(maxsize=60)
control_lock = threading.RLock()

imu = MPU6050(bus_number=1)


# ---------- HARDWARE ----------

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

motor_updated = [0.0, 0.0]


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

    for index, motor in enumerate(motors):
        if m is motor:
            motor_updated[index] = time.monotonic()


def sonar_ranges():
    return S["dist"][FRONT_SONAR], S["dist"][REAR_SONAR]


def obstacle_blocks(l, r, front_cm, rear_cm, stop_cm, enabled):
    if not enabled:
        return []

    def near(d):
        return (
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


# ---------- MINI MAP V0 ----------

class MiniMap:
    # Means from the six real trials at 40% motor power.
    # Provisional extrapolation beyond the original 1-second trials.
    SPEEDS = {
        "forward": 21.1087887108,
        "reverse": 22.0637781360,
    }

    def __init__(self):
        self.origin = None
        self.valid = False
        self.reason = "Calibrate IMU, then Reset map"

        self.x = 0.0
        self.y = 0.0
        self.heading = 0.0
        self.distance = 0.0

        self.points = []
        self.previous_time = None
        self.previous_mode = "stop"
        self.previous_heading = 0.0
        self.started = None

    @staticmethod
    def imu_ready(data):
        yaw = data.get("yaw_deg")
        return (
            data.get("valid")
            and data.get("yaw_valid")
            and isinstance(yaw, (int, float))
            and math.isfinite(yaw)
            and data.get("calibration") != "running"
        )

    def invalidate(self, reason):
        if self.valid:
            self.valid = False
            self.reason = reason + "; Reset map required"

    def reset(self, data):
        if not self.imu_ready(data):
            raise ValueError("Fresh calibrated IMU angle required")

        self.__init__()
        self.origin = data["yaw_deg"]
        self.valid = True
        self.reason = "Recording approximate path at 40% power"
        self.points = [[0.0, 0.0, "start"]]
        self.started = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    @staticmethod
    def mode(a, b):
        if not all(math.isfinite(v) for v in (a, b)):
            return "unsupported"

        if max(abs(a), abs(b)) < 0.001:
            return "stop"

        # Pivot rotation: translation is assumed zero.
        if a * b < 0 and abs(a + b) < 0.002:
            return "pivot"

        if abs(a - 0.4) < 0.002 and abs(b - 0.4) < 0.002:
            return "forward"

        if abs(a + 0.4) < 0.002 and abs(b + 0.4) < 0.002:
            return "reverse"

        return "unsupported"

    def step(self, now, data, applied, testing, output_fresh):
        if not self.valid:
            return

        if testing:
            self.invalidate("Hardware test interrupted mapping")
            return

        if not self.imu_ready(data):
            self.invalidate("IMU angle unavailable")
            return

        if not output_fresh:
            self.invalidate("Motor control updates unavailable")
            return

        current = self.mode(*applied)

        if current == "unsupported":
            self.invalidate(
                "Use separate arrows at 40%, motor trims 1.0"
            )
            return

        heading = data["yaw_deg"] - self.origin

        if self.previous_time is not None:
            dt = now - self.previous_time

            if not 0 < dt <= 0.25:
                self.invalidate("Mapping sample gap")
                return

            # Integrate the command active over the preceding interval.
            mode = self.previous_mode

            if mode in self.SPEEDS:
                if len(self.points) >= 12000:
                    self.invalidate("Path limit reached; export JSON")
                    return

                distance = self.SPEEDS[mode] * dt
                angle = math.radians(
                    (self.previous_heading + heading) / 2
                )
                signed = (
                    distance if mode == "forward" else -distance
                )

                self.x += signed * math.cos(angle)
                self.y += signed * math.sin(angle)
                self.distance += distance
                self.points.append([self.x, self.y, mode])

        self.heading = heading
        self.previous_heading = heading
        self.previous_time = now
        self.previous_mode = current

    def snapshot(self):
        return dict(
            schema=1,
            valid=self.valid,
            reason=self.reason,
            started=self.started,
            x_cm=self.x,
            y_cm=self.y,
            heading_deg=self.heading,
            distance_cm=self.distance,
            points=self.points,
            speeds_cm_s=self.SPEEDS,
            assumptions=(
                "40% power, level tile floor; old 1 s trials "
                "extrapolated. Relative gyro heading; pivot "
                "translation assumed zero. No encoders, "
                "slip/coasting correction or SLAM. Reverse is "
                "less reliable. RAM only; export before restart."
            ),
        )


minimap = MiniMap()
map_stop = threading.Event()


def map_loop():
    while not map_stop.wait(0.02):
        with control_lock:
            now = time.monotonic()

            try:
                minimap.step(
                    now,
                    imu.snapshot(),
                    S["applied_drive"],
                    S["testing"],
                    all(m is not None for m in motors)
                    and all(
                        0 <= now - t < 0.2
                        for t in motor_updated
                    ),
                )
            except Exception as error:
                minimap.invalidate(
                    "Map error: " + str(error)[:100]
                )


@app.get("/api/map/status")
def api_map_status():
    with control_lock:
        return jsonify(minimap.snapshot())


@app.post("/api/map/reset")
def api_map_reset():
    with control_lock:
        if (
            S["testing"]
            or S.get("motion_test_pending", 0)
            or any(
                abs(v) > 0.001
                for v in S["applied_drive"]
            )
            or S["l"]
            or S["r"]
        ):
            return jsonify(
                ok=False,
                msg="Stop the robot first",
            ), 409

        if any(m is None for m in motors):
            return jsonify(
                ok=False,
                msg="Both motors must be available",
            ), 409

        try:
            minimap.reset(imu.snapshot())
        except ValueError as error:
            return jsonify(
                ok=False,
                msg=str(error),
            ), 409

        return jsonify(ok=True, msg="Map origin reset")


@app.get("/api/map/export")
def api_map_export():
    with control_lock:
        payload = json.dumps(
            minimap.snapshot(),
            ensure_ascii=False,
            allow_nan=False,
        )

    return Response(
        payload,
        mimetype="application/json",
        headers={
            "Content-Disposition":
                'attachment; filename="mini_map_v0.json"'
        },
    )


# ---------- CAMERA / DETECTION ----------

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

        return YOLO(
            os.path.join(HERE, ncnn),
            task="detect",
        )

    except Exception as e:
        print("NCNN unavailable, using PyTorch:", e)
        return YOLO(os.path.join(HERE, f"{MODEL}.pt"))


def detect_loop():
    model = None
    streak = {c: 0 for c in CLASSES.values()}
    n, t0 = 0, time.time()

    while True:
        if not S["detect"] or frame["jpg"] is None:
            S["alert"] = []
            n, t0 = 0, time.time()
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

            for box in res.boxes:
                name = CLASSES[int(box.cls)]
                seen.add(name)

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                col = COLORS[name]
                txt = f"{name.capitalize()} {float(box.conf):.0%}"

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
                c for c, count in streak.items()
                if count >= CFG["confirm"]
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
                S["det_fps"] = round(
                    n / (time.time() - t0), 1
                )
                n, t0 = 0, time.time()

        except Exception as e:
            S["det_state"] = f"error: {str(e)[:90]}"
            S["detect"] = False
            model = None
            print("Detection error:", e)


# ---------- SYSTEM / AUDIO ----------

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
    with open("/proc/meminfo") as f:
        m = {
            line.split(":")[0]: int(line.split()[1])
            for line in f
            if ":" in line
        }

    memory = round(
        100 * (1 - m["MemAvailable"] / m["MemTotal"])
    )
    cpu = round(os.getloadavg()[0] / os.cpu_count() * 100)
    return memory, cpu


def beep(path, freq=440, dur=1.2, rate=48000):
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rate)

        values = (
            int(12000 * math.sin(2 * math.pi * freq * i / rate))
            for i in range(int(rate * dur))
        )
        w.writeframes(
            b"".join(struct.pack("<hh", v, v) for v in values)
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
        (c for c in cards if "usb" in (c[1] + c[2]).lower()),
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
            "no microphone found (arecord -l is empty): "
            "plug in the USB mic",
        )

    result = subprocess.run(
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

    return path, result


def levels(path):
    with wave.open(path) as w:
        samples = array.array("h", w.readframes(w.getnframes()))

    def db(v):
        return 20 * math.log10(max(v, 1) / 32768)

    rms = math.sqrt(
        sum(x * x for x in samples) / max(len(samples), 1)
    )
    peak = max((abs(x) for x in samples), default=0)

    return db(rms), db(peak)


# ---------- MODULE TESTS ----------

def t_sonar():
    d = sonar_ranges()
    ok = all(x is not None and 2 < x < 299 for x in d)
    msg = f"Front {d[0]} cm, rear {d[1]} cm"

    if not ok:
        msg += " (no echo: check Echo wire, divider, 5V)"

    return ok, msg


def t_camera():
    jpg = frame["jpg"]
    msg = (
        f"frame {len(jpg) // 1024} KB, {S['fps']} fps"
        if jpg
        else "no frames (check ribbon cable / picamera2)"
    )
    return bool(jpg), msg


def t_speaker():
    path = os.path.join(tempfile.gettempdir(), "beep.wav")
    beep(path)
    result = play(path)

    msg = (
        "440 Hz tone played, did you hear it?"
        if result.returncode == 0
        else result.stderr.strip()[:120]
    )
    return result.returncode == 0, msg


def t_mic():
    path, result = record(2)

    if result.returncode != 0:
        return False, result.stderr.strip()[:120] or "arecord failed"

    rms, peak = levels(path)
    msg = f"level {rms:.0f} dB, peak {peak:.0f} dB"

    if peak <= -40:
        msg += " (too quiet: speak or clap while recording)"

    return peak > -40, msg


def t_echo():
    path, result = record(3)

    if result.returncode != 0:
        return False, "recording failed"

    return (
        play(path).returncode == 0,
        "recorded 3 s and played it back",
    )


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
    imu=imu.test,
)


def talk_worker():
    player = None

    while True:
        chunk = talk_q.get()

        if chunk is None:
            if player:
                try:
                    player.stdin.close()
                    player.wait(timeout=3)
                except Exception:
                    player.kill()

            player = None
            continue

        if player is None or player.poll() is not None:
            player = subprocess.Popen(
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
            player.stdin.write(chunk)
            player.stdin.flush()

        except Exception as e:
            err = (
                player.stderr.read().decode(errors="ignore")[:160]
                if player.poll() is not None
                else ""
            )
            S["talk_err"] = (err or str(e)).strip()
            print("talk: speaker error:", S["talk_err"])
            player = None


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
                jpg = out["jpg"]

            if jpg:
                yield (
                    b"--f\r\nContent-Type: image/jpeg\r\n\r\n"
                    + jpg + b"\r\n"
                )

    return Response(
        gen(),
        mimetype="multipart/x-mixed-replace; boundary=f",
    )


@app.post("/api/drive")
def api_drive():
    j = request.get_json(force=True)

    with control_lock:
        if imu.is_calibrating():
            return jsonify(
                ok=False,
                msg="IMU calibration running",
            ), 409

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


@app.post("/api/imu/calibrate")
def api_imu_calibrate():
    with control_lock:
        if S["testing"]:
            return jsonify(
                ok=False,
                msg="Wait for hardware test to finish",
            ), 409

        ok, msg = imu.start_calibration()

        if ok:
            minimap.invalidate("IMU recalibrated")
            S.update(
                l=0.0,
                r=0.0,
                t=0.0,
                applied_drive=[0.0, 0.0],
            )

            for m in motors:
                drive(m, 0)

    return jsonify(ok=ok, msg=msg), (202 if ok else 409)


@app.post("/api/imu/zero")
def api_imu_zero():
    with control_lock:
        if (
            S["testing"]
            or any(S["applied_drive"])
            or S["l"]
            or S["r"]
        ):
            return jsonify(
                ok=False,
                msg="Stop the robot first",
            ), 409

        ok, msg = imu.zero_yaw()

        if ok:
            minimap.invalidate("IMU angle reset")

    return jsonify(ok=ok, msg=msg), (200 if ok else 409)


@app.post("/api/safety")
def api_safety():
    S["safety"] = bool(
        request.get_json(force=True).get("on", True)
    )
    return jsonify(ok=True)


@app.post("/api/detect")
def api_detect():
    S["detect"] = bool(
        request.get_json(force=True).get("on", False)
    )
    S["det_state"] = "starting..." if S["detect"] else "off"
    return jsonify(ok=True)


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

    temperature = (
        sh(["vcgencmd", "measure_temp"])
        .replace("temp=", "")
        .replace("'C", "")
    )
    front_cm, rear_cm = sonar_ranges()

    return jsonify(
        imu=imu.snapshot(),
        dist=S["dist"],
        front_cm=front_cm,
        rear_cm=rear_cm,
        blocked_by=S["blocked_by"],
        applied_drive=S["applied_drive"],
        blocked=S["blocked"],
        safety=S["safety"],
        fps=S["fps"],
        cam=S["cam"],
        temp=temperature,
        throttled=sh(
            ["vcgencmd", "get_throttled"]
        ).replace("throttled=", ""),
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

    if name == "imu":
        ok, msg = imu.test()
        return jsonify(ok=ok, msg=msg)

    with control_lock:
        if S["testing"] or imu.is_calibrating():
            return jsonify(
                ok=False,
                msg="Test or calibration already running",
            ), 409

        minimap.invalidate("Hardware test started")
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

    with open(CFG_FILE, "w") as f:
        json.dump(CFG, f, indent=2)

    return jsonify(ok=True, cfg=CFG)


# ---------- STARTUP / CLEANUP ----------

def cleanup():
    map_stop.set()

    for m in motors:
        if m:
            m.stop()

    imu.close()


def https_context():
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


# Connect the existing motion module directly.
# It provides control(), sonar_loop() and /api/motion/* routes.
from motion import install as install_motion
install_motion(globals())

from pilot import install as install_pilot
install_pilot(globals())


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    tls = https_context()
    atexit.register(cleanup)
    imu.start()

    for worker in (
        control,
        sonar_loop,
        camera_loop,
        detect_loop,
        talk_worker,
        map_loop,
    ):
        threading.Thread(target=worker, daemon=True).start()

    app.run(
        host="0.0.0.0",
        port=8000,
        threaded=True,
        ssl_context=tls,
    )