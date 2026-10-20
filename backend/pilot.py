"""SEE pilot recording and a persistent, preloaded YOLO baseline."""
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import statistics
import subprocess
import threading
import time
import uuid
import zipfile

from flask import jsonify, request, send_file


def write_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def percentile(values, fraction):
    if not values:
        return None

    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def hardware_stats():
    result = {"unix_s": time.time()}

    try:
        memory = dict(
            (line.split(":")[0], int(line.split()[1]))
            for line in Path("/proc/meminfo").read_text().splitlines()
        )

        result["system_used_mb"] = (
            memory["MemTotal"] - memory["MemAvailable"]
        ) / 1024

        status = Path("/proc/self/status").read_text()
        result["server_rss_mb"] = (
            int(re.search(r"VmRSS:\s+(\d+)", status)[1]) / 1024
        )
        result["load_1min"] = os.getloadavg()[0]

        result["temp_c"] = float(
            Path(
                "/sys/class/thermal/thermal_zone0/temp"
            ).read_text()
        ) / 1000

    except (OSError, ValueError, TypeError):
        pass

    try:
        result["throttled"] = subprocess.run(
            ["vcgencmd", "get_throttled"],
            capture_output=True,
            text=True,
            timeout=0.5,
        ).stdout.strip()

    except (OSError, subprocess.SubprocessError):
        result["throttled"] = "unavailable"

    return result


class Recorder:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.run = None
        self.latest = None
        self.export = dict(state="idle", id=None, done=0, total=0, error="")
        self.export_path = None
        self.export_signature = None
        self.export_purpose = None

    def busy(self):
        with self.lock:
            return self.run is not None or self.export["state"] == "building"

    def status(self):
        with self.lock:
            r = self.run

            return dict(
                state=r["state"] if r else "idle",
                id=r["id"] if r else None,
                elapsed_s=(
                    round(time.monotonic() - r["start"], 1)
                    if r else 0
                ),
                dropped=r["dropped"] if r else 0,
                latest=self.latest,
                recordings=self.recordings(),
                export=dict(self.export),
            )

    def start(self, data, metadata):
        scene = str(data.get("scene", "")).strip()

        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", scene):
            raise ValueError(
                "Scene: use 1–40 letters, digits, _ or -"
            )

        purpose = data.get("purpose", "pilot")
        if purpose not in ("pilot", "yolo_dataset"):
            raise ValueError("Unknown recording purpose")
        dataset = None
        if purpose == "yolo_dataset":
            dataset = data.get("dataset")
            if not isinstance(dataset, dict):
                raise ValueError("Dataset metadata is required")
            for key in ("subject", "session", "pose", "lighting", "distance_m", "take"):
                value = str(dataset.get(key, "")).strip()
                if not value or len(value) > 80:
                    raise ValueError("Required dataset field: " + key)
            if dataset["pose"] not in ("standing", "sitting", "crouching", "lying", "occluded", "negative"):
                raise ValueError("Unknown pose")
            if dataset["lighting"] not in ("normal", "dim", "backlit"):
                raise ValueError("Unknown lighting")
            distance = float(dataset["distance_m"])
            take = int(dataset["take"])
            if not math.isfinite(distance) or not 0 <= distance <= 20 or not 1 <= take <= 100:
                raise ValueError("Invalid distance or take")
            dataset = {key: str(dataset[key]).strip() for key in
                       ("subject", "session", "pose", "lighting", "distance_m", "take")}
            dataset.update(split="unassigned", annotation_status="unlabelled")
        seconds = float(data.get("seconds", 15))

        if not math.isfinite(seconds) or not 5 <= seconds <= 30:
            raise ValueError("Duration: 5–30 seconds")

        if shutil.disk_usage(self.root).free < 1024**3:
            raise ValueError(
                "At least 1 GB of free disk space required"
            )

        with self.lock:
            if self.run or self.export["state"] == "building":
                raise ValueError(
                    "Recording/finalization already running"
                )

            identity = (
                time.strftime("%Y%m%d_%H%M%S")
                + "_" + scene + "_"
                + uuid.uuid4().hex[:8]
            )

            folder = self.root / identity
            folder.mkdir()
            (folder / "frames").mkdir()

            metadata.update(
                id=identity,
                scene=scene,
                purpose=purpose,
                dataset=dataset,
                requested_seconds=seconds,
                started_utc=time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),
                description=str(
                    data.get("description", "")
                )[:1000],
                expected=str(
                    data.get("expected", "unspecified")
                )[:50],
                recording=dict(
                    format="MJPEG AVI",
                    nominal_fps=10,
                    size=[640, 480],
                ),
            )

            write_json(folder / "metadata.json", metadata)

            now = time.monotonic()

            r = dict(
                id=identity,
                folder=folder,
                meta=metadata,
                start=now,
                end=now + seconds,
                state="recording",
                reason="duration reached",
                q=queue.Queue(maxsize=64),
                next_frame=now,
                inflight=0,
                dropped=0,
                error="",
                stop=threading.Event(),
            )

            self.run = r

            thread = threading.Thread(
                target=self.worker,
                args=(r,),
                daemon=True,
            )
            r["thread"] = thread
            thread.start()

            return identity

    def recordings(self):
        rows = []
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir() or folder.is_symlink() or folder.name.startswith('_'):
                continue
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', folder.name):
                continue
            meta_path = folder / 'metadata.json'
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
                report_path = folder / 'report.json'
                report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.is_file() else None
                warnings = report.get('warnings', []) if report else ['Report missing; possibly interrupted recording']
                rows.append(dict(
                    id=folder.name, scene=meta.get('scene', '?'),
                    purpose=meta.get('purpose', 'pilot'), dataset=meta.get('dataset'),
                    expected=meta.get('expected', '?'),
                    seconds=report.get('interval_seconds') if report else None,
                    predictions=report.get('successful_predictions') if report else None,
                    person_frames=report.get('frames_with_person_prediction') if report else None,
                    warnings=warnings,
                    status='reported' if report else 'incomplete',
                    report_mtime=report_path.stat().st_mtime_ns if report else None,
                ))
            except (OSError, ValueError, TypeError, AttributeError) as error:
                rows.append(dict(id=folder.name, scene='?', expected='?', seconds=None,
                                 predictions=None, person_frames=None, status='unreadable',
                                 report_mtime=None, warnings=[str(error)]))
        return rows

    def export_all(self, purpose=None):
        with self.lock:
            if self.run:
                raise ValueError('Finish the current recording first')
            if self.export['state'] == 'building':
                if purpose != self.export_purpose:
                    raise ValueError('Another archive is building; wait for completion')
                return dict(self.export)
            rows = self.recordings()
            if purpose:
                rows = [r for r in rows if r.get("purpose") == purpose]
            if not rows:
                raise ValueError('No saved recordings found')
            signature = [(r['id'], r['report_mtime'], r['status']) for r in rows]
            if (self.export['state'] == 'ready' and self.export_signature == signature
                    and self.export_path and self.export_path.is_file()):
                return dict(self.export)
            identity = uuid.uuid4().hex
            self.export = dict(state='building', id=identity, done=0, total=len(rows), error='', purpose=purpose)
            self.export_signature = signature
            self.export_purpose = purpose
            threading.Thread(target=self._export_worker, args=(rows, identity), daemon=True).start()
            return dict(self.export)

    def _export_worker(self, rows, identity):
        destination = self.root / '_exports'
        temp = destination / (identity + '.tmp')
        final = destination / (identity + '.zip')
        try:
            destination.mkdir(exist_ok=True)
            entries = []
            for row in rows:
                folder = self.root / row['id']
                if folder.is_symlink():
                    raise ValueError('Recording folder changed during export')
                files = []
                for path in sorted(folder.rglob('*')):
                    relative = path.relative_to(folder)
                    if (path.is_symlink() or any(p.is_symlink() for p in path.parents if p != self.root)
                            or not path.is_file() or path.name == 'bundle.zip'
                            or path.suffix == '.tmp'):
                        continue
                    files.append((path, row['id'] + '/' + relative.as_posix()))
                entries.append(files)
            needed = sum(p.stat().st_size for files in entries for p, _ in files)
            if shutil.disk_usage(destination).free < needed + 100*1024**2:
                raise ValueError('Not enough free space for the combined ZIP')
            manifest = dict(
                schema=1, recording_count=len(rows), recordings=rows,
                purpose=self.export_purpose,
                note='Selected stored recordings; interrupted records are included and flagged. '
                     'Individual bundle.zip files omitted to avoid duplicate data. '
                     'Original metadata is preserved; expected labels are not automatically corrected.'
            )
            with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as z:
                z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
                for index, files in enumerate(entries):
                    for path, name in files:
                        z.write(path, name)
                    with self.lock:
                        self.export['done'] = index + 1
            os.replace(temp, final)
            with self.lock:
                self.export_path = final
                self.export.update(state='ready', bytes=final.stat().st_size)
        except Exception as error:
            # Only our incomplete export file is removed; scene data is untouched.
            if temp.exists():
                temp.unlink()
            with self.lock:
                self.export.update(state='error', error=str(error))


    def stop(self, reason="operator STOP"):
        with self.lock:
            if self.run and not self.run["stop"].is_set():
                self.run["end"] = min(
                    self.run["end"],
                    time.monotonic(),
                )
                self.run["reason"] = reason
                self.run["stop"].set()

    def camera(self, seq, stamp, jpg):
        with self.lock:
            r = self.run

            if (
                not r
                or r["state"] != "recording"
                or r["stop"].is_set()
                or time.monotonic() >= r["end"]
                or not r["start"] <= stamp < r["end"]
            ):
                return

            if stamp < r["next_frame"]:
                return

            r["next_frame"] = stamp + 0.1

            try:
                r["q"].put_nowait(
                    ("frame", seq, stamp, jpg, None)
                )
            except queue.Full:
                r["dropped"] += 1

    def begin_prediction(self, stamp):
        with self.lock:
            r = self.run

            if (
                r
                and r["state"] == "recording"
                and not r["stop"].is_set()
                and time.monotonic() < r["end"]
                and r["start"] <= stamp < r["end"]
            ):
                r["inflight"] += 1
                return r["id"]

        return None

    def prediction(self, identity, seq, stamp, jpg, result):
        if identity is None:
            return

        with self.lock:
            r = self.run

            if not r or r["id"] != identity:
                return

            r["inflight"] -= 1

            try:
                r["q"].put_nowait(
                    ("prediction", seq, stamp, jpg, result)
                )
            except queue.Full:
                r["dropped"] += 1

    def worker(self, r):
        folder = r["folder"]
        writer = None

        video_rows = []
        predictions = []
        telemetry = []

        drain_deadline = None
        next_stats = 0
        files = []

        try:
            import cv2
            import numpy as np

            writer = cv2.VideoWriter(
                str(folder / "raw.avi"),
                cv2.VideoWriter_fourcc(*"MJPG"),
                10,
                (640, 480),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    "OpenCV MJPG video writer unavailable"
                )

            frames_file = open(
                folder / "frames.jsonl",
                "w",
                encoding="utf-8",
            )
            preds_file = open(
                folder / "predictions.jsonl",
                "w",
                encoding="utf-8",
            )
            stats_file = open(
                folder / "telemetry.jsonl",
                "w",
                encoding="utf-8",
            )
            files = [frames_file, preds_file, stats_file]

            while True:
                now = time.monotonic()

                with self.lock:
                    if (
                        now >= r["end"]
                        or r["stop"].is_set()
                    ) and drain_deadline is None:
                        r["state"] = "finalizing"
                        drain_deadline = now + 10

                    done = (
                        drain_deadline is not None
                        and (
                            (
                                r["q"].empty()
                                and r["inflight"] == 0
                            )
                            or now >= drain_deadline
                        )
                    )

                if done:
                    break

                if now >= next_stats:
                    sample = hardware_stats()
                    sample["t_s"] = now - r["start"]

                    telemetry.append(sample)
                    stats_file.write(
                        json.dumps(sample) + "\n"
                    )
                    stats_file.flush()

                    next_stats = now + 1

                    if (
                        shutil.disk_usage(folder).free
                        < 100 * 1024**2
                    ):
                        raise RuntimeError(
                            "Disk space below 100 MB"
                        )

                try:
                    kind, seq, stamp, jpg, result = (
                        r["q"].get(timeout=0.05)
                    )
                except queue.Empty:
                    continue

                image_path = (
                    folder / "frames" / f"{seq:09d}.jpg"
                )

                if not image_path.exists():
                    image_path.write_bytes(jpg)

                row = dict(
                    seq=seq,
                    t_s=stamp - r["start"],
                    image=f"frames/{seq:09d}.jpg",
                )

                if kind == "frame":
                    img = cv2.imdecode(
                        np.frombuffer(jpg, np.uint8),
                        cv2.IMREAD_COLOR,
                    )

                    if (
                        img is None
                        or img.shape[:2] != (480, 640)
                    ):
                        raise RuntimeError(
                            "Invalid camera frame or dimensions"
                        )

                    writer.write(img)
                    row["video_index"] = len(video_rows)
                    video_rows.append(row)

                    frames_file.write(
                        json.dumps(row) + "\n"
                    )
                    frames_file.flush()

                else:
                    row.update(result)
                    predictions.append(row)

                    preds_file.write(
                        json.dumps(
                            row,
                            allow_nan=False,
                        ) + "\n"
                    )
                    preds_file.flush()

        except Exception as error:
            r["error"] = str(error)
            r["end"] = min(
                r["end"],
                time.monotonic(),
            )

        finally:
            with self.lock:
                r["state"] = "finalizing"

            if writer is not None:
                writer.release()

            for f in files:
                f.close()

        try:
            report = self.report(
                r,
                video_rows,
                predictions,
                telemetry,
            )

            write_json(folder / "report.json", report)

            text = "# SEE pilot report\n\n"
            text += (
                "Operational pilot measurements; "
                "NOT an accuracy evaluation.\n\n"
            )
            text += (
                "```json\n"
                + json.dumps(
                    report,
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n```\n"
            )

            (folder / "report.md").write_text(
                text,
                encoding="utf-8",
            )

            archive = folder / "bundle.zip"

            with zipfile.ZipFile(
                archive,
                "w",
                compression=zipfile.ZIP_STORED,
            ) as z:
                for path in sorted(folder.rglob("*")):
                    if path.is_file() and path != archive:
                        z.write(
                            path,
                            path.relative_to(folder),
                        )

            latest = dict(
                id=r["id"],
                ok=not r["error"],
                report=report,
            )

        except Exception as error:
            latest = dict(
                id=r["id"],
                ok=False,
                error="Report/export failed: " + str(error),
            )

        with self.lock:
            self.latest = latest
            self.run = None

    @staticmethod
    def report(r, frames, predictions, telemetry):
        good = [
            p for p in predictions
            if not p.get("error")
        ]
        times = [
            p["predict_wall_ms"]
            for p in good
        ]

        duration = max(0, r["end"] - r["start"])
        detected = sum(
            bool(p["boxes"])
            for p in good
        )

        warnings = []

        if r["error"]:
            warnings.append(r["error"])

        if r["dropped"]:
            warnings.append(
                "Recorder queue overflow; records missing"
            )

        if r["inflight"]:
            warnings.append(
                "Inference did not finish within drain timeout"
            )

        if not frames:
            warnings.append("No video frames saved")

        if len(frames) < duration * 8:
            warnings.append(
                "Video sampling below 8 fps"
            )

        if not good and r["meta"].get("baseline_enabled", True):
            warnings.append(
                "No successful baseline predictions"
            )

        if r["reason"] not in (
            "duration reached",
            "operator STOP",
        ):
            warnings.append(r["reason"])

        if len(good) != len(predictions):
            warnings.append(
                "Some predictions failed"
            )

        if any(
            s.get("throttled") not in (
                None,
                "",
                "unavailable",
                "throttled=0x0",
            )
            for s in telemetry
        ):
            warnings.append(
                "Nonzero throttling flags; "
                "may include historical events"
            )

        def peak(key):
            values = [
                s[key]
                for s in telemetry
                if key in s
            ]
            return max(values) if values else None

        return dict(
            id=r["id"],
            scene=r["meta"]["scene"],
            purpose=r["meta"].get("purpose", "pilot"),
            dataset=r["meta"].get("dataset"),
            expected=r["meta"]["expected"],
            description=r["meta"]["description"],
            stop_reason=r["reason"],
            interval_seconds=duration,
            video_frames=len(frames),
            video_nominal_fps=10,
            saved_video_fps=(
                len(frames) / duration
                if duration else None
            ),
            successful_predictions=len(good),
            prediction_errors=(
                len(predictions) - len(good)
            ),
            processed_camera_fps=(
                len(good) / duration
                if duration else None
            ),
            frames_with_person_prediction=detected,
            person_prediction_frame_fraction=(
                detected / len(good)
                if good else None
            ),
            predict_wall_ms_median=(
                statistics.median(times)
                if times else None
            ),
            predict_wall_ms_p95=percentile(
                times,
                0.95,
            ),
            sampled_peak_server_rss_mb=peak(
                "server_rss_mb"
            ),
            sampled_peak_system_used_mb=peak(
                "system_used_mb"
            ),
            sampled_max_temp_c=peak("temp_c"),
            queue_drops=r["dropped"],
            unfinished_predictions=r["inflight"],
            warnings=warnings,
            accuracy_metrics=None,
            interpretation=(
                "Prediction frequency is NOT recall or precision. "
                "Manual boxes are required for precision/recall/AP. "
                "No count of unique people. Timings include "
                "preprocess+inference+postprocess inside model.predict, "
                "not JPEG decoding, drawing, disk writes or browser "
                "latency. AVI plays at fixed 10 fps; frames.jsonl "
                "contains actual host capture-completion timestamps. "
                "frames/ also contains exact inputs used by recorded "
                "predictions. Recorder and dashboard were active; "
                "warm-up excluded. RAM peaks are sampled."
            ),
        )


def install(g):
    app = g["app"]
    state = g["S"]
    cfg = g["CFG"]
    root = Path(g["HERE"])

    recorder = Recorder(
        root / "data" / "see_pilot"
    )

    info = dict(
        ready=False,
        state="loading",
        error="",
        model="yolo11n",
        imgsz=320,
    )
    info_lock = threading.Lock()

    state["detect"] = True
    state["det_state"] = "loading baseline..."
    g["frame"].update(seq=0, stamp=0)

    app.add_url_rule("/dataset", "dataset_page",
                     lambda: send_file(root / "dashboard" / "dataset.html"))

    def model_info():
        with info_lock:
            return copy.deepcopy(info)

    def camera_loop():
        cam = None

        try:
            from picamera2 import Picamera2
            from libcamera import Transform
            import io

            cam = Picamera2()
            cam.configure(
                cam.create_video_configuration(
                    main={"size": (640, 480)},
                    transform=Transform(
                        hflip=int(cfg["flip_camera"]),
                        vflip=int(cfg["flip_camera"]),
                    ),
                )
            )
            cam.options["quality"] = 70
            cam.start()

            state["cam"] = True
            seq = 0
            count = 0
            since = time.monotonic()

            while True:
                buf = io.BytesIO()
                cam.capture_file(buf, format="jpeg")

                stamp = time.monotonic()
                jpg = buf.getvalue()
                seq += 1

                recorder.camera(seq, stamp, jpg)

                with g["cond"]:
                    g["frame"].update(
                        jpg=jpg,
                        seq=seq,
                        stamp=stamp,
                    )

                    if (
                        not state["detect"]
                        or not model_info()["ready"]
                    ):
                        g["out"]["jpg"] = jpg

                    g["cond"].notify_all()

                count += 1

                if stamp - since >= 2:
                    state["fps"] = round(
                        count / (stamp - since),
                        1,
                    )
                    count = 0
                    since = stamp

        except Exception as error:
            state["cam"] = False
            recorder.stop(
                "camera failed: " + str(error)
            )
            print(
                "Camera error:",
                error,
                flush=True,
            )

        finally:
            if cam is not None:
                cam.close()

    def detect_loop():
        try:
            import cv2
            import numpy as np
            from ultralytics import YOLO

            models = root / "models"
            models.mkdir(exist_ok=True)

            candidates = [
                models / "yolo11n_ncnn_model",
                root / "yolo11n_ncnn_model",
                models / "yolo11n.pt",
                root / "yolo11n.pt",
            ]

            errors = []
            model = None

            for path in candidates:
                if not path.exists():
                    continue

                try:
                    candidate = YOLO(
                        str(path),
                        task="detect",
                    )

                    for _ in range(3):
                        candidate.predict(
                            np.zeros(
                                (480, 640, 3),
                                dtype=np.uint8,
                            ),
                            imgsz=320,
                            classes=[0],
                            device="cpu",
                            verbose=False,
                        )

                    model = candidate
                    break

                except Exception as error:
                    errors.append(
                        str(path) + ": " + str(error)
                    )

            if model is None:
                if any(
                    p.exists()
                    for p in candidates
                ):
                    raise RuntimeError(
                        "Local models failed: "
                        + " | ".join(errors)
                    )

                path = models / "yolo11n.pt"

                # Downloaded once by Ultralytics if absent.
                model = YOLO(
                    str(path),
                    task="detect",
                )

                for _ in range(3):
                    model.predict(
                        np.zeros(
                            (480, 640, 3),
                            dtype=np.uint8,
                        ),
                        imgsz=320,
                        classes=[0],
                        device="cpu",
                        verbose=False,
                    )

            hashes = {}

            model_files = (
                [path]
                if path.is_file()
                else sorted(path.rglob("*"))
            )

            for file in model_files:
                if file.is_file():
                    h = hashlib.sha256()

                    with file.open("rb") as f:
                        for chunk in iter(
                            lambda: f.read(1024 * 1024),
                            b"",
                        ):
                            h.update(chunk)

                    hashes[
                        str(file.relative_to(root))
                    ] = h.hexdigest()

            with info_lock:
                info.update(
                    ready=True,
                    state="ready",
                    path=str(path.relative_to(root)),
                    format=(
                        "NCNN"
                        if path.is_dir()
                        else "PyTorch"
                    ),
                    sha256=hashes,
                    python=platform.python_version(),
                    platform=platform.platform(),
                    ultralytics=importlib.metadata.version(
                        "ultralytics"
                    ),
                    opencv=cv2.__version__,
                    fallback_notes=errors,
                    warmup_calls=3,
                )

        except Exception as error:
            with info_lock:
                info.update(
                    ready=False,
                    state="error",
                    error=str(error),
                )

            state.update(
                detect=False,
                det_state=(
                    "baseline error: "
                    + str(error)[:120]
                ),
            )

            print(
                "Baseline load error:",
                error,
                flush=True,
            )
            return

        last_seq = -1
        count = 0
        since = time.monotonic()
        streak = 0

        while True:
            if not state["detect"]:
                state.update(
                    det_state="ready / off",
                    alert=[],
                    det_fps=0,
                )
                streak = 0
                count = 0
                since = time.monotonic()
                time.sleep(0.1)
                continue

            with g["cond"]:
                g["cond"].wait_for(
                    lambda: (
                        g["frame"]["seq"] != last_seq
                    ),
                    timeout=0.5,
                )
                item = dict(g["frame"])

            if (
                item["jpg"] is None
                or item["seq"] == last_seq
            ):
                continue

            last_seq = item["seq"]

            if (
                time.monotonic() - item["stamp"]
                > 0.5
            ):
                continue

            token = recorder.begin_prediction(
                item["stamp"]
            )
            result = {
                "error": "prediction not completed"
            }

            try:
                img = cv2.imdecode(
                    np.frombuffer(
                        item["jpg"],
                        np.uint8,
                    ),
                    cv2.IMREAD_COLOR,
                )

                started = time.monotonic()

                pred = model.predict(
                    img,
                    classes=[0],
                    conf=float(cfg["det_conf"]),
                    imgsz=320,
                    iou=0.7,
                    max_det=100,
                    device="cpu",
                    verbose=False,
                )[0]

                wall_ms = (
                    time.monotonic() - started
                ) * 1000

                boxes = [
                    dict(
                        xyxy=[
                            float(v)
                            for v in b.xyxy[0].tolist()
                        ],
                        confidence=float(
                            b.conf.item()
                        ),
                        class_id=0,
                    )
                    for b in pred.boxes
                ]

                result = dict(
                    boxes=boxes,
                    predict_wall_ms=wall_ms,
                    library_speed_ms=pred.speed,
                    frame_age_at_result_ms=(
                        time.monotonic() - item["stamp"]
                    ) * 1000,
                )

                streak = streak + 1 if boxes else 0

                state.update(
                    see_result_at=item["stamp"],
                    det_state="baseline running",
                    infer_ms=round(wall_ms),
                    alert=(
                        ["person"]
                        if streak >= cfg["confirm"]
                        else []
                    ),
                )

                for b in boxes:
                    x1, y1, x2, y2 = map(
                        int,
                        b["xyxy"],
                    )

                    cv2.rectangle(
                        img,
                        (x1, y1),
                        (x2, y2),
                        (176, 198, 60),
                        2,
                    )

                    cv2.putText(
                        img,
                        f"person {b['confidence']:.0%}",
                        (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (176, 198, 60),
                        2,
                    )

                with g["cond"]:
                    if state["detect"]:
                        g["out"]["jpg"] = (
                            cv2.imencode(
                                ".jpg",
                                img,
                            )[1].tobytes()
                        )
                        g["cond"].notify_all()

                count += 1
                now = time.monotonic()

                if now - since >= 2:
                    state["det_fps"] = round(
                        count / (now - since),
                        1,
                    )
                    count = 0
                    since = now

            except Exception as error:
                result = dict(error=str(error))

                state.update(
                    detect=False,
                    alert=[],
                    det_state=(
                        "baseline error: "
                        + str(error)[:100]
                    ),
                )
                recorder.stop(
                    "baseline prediction error"
                )

            finally:
                recorder.prediction(
                    token,
                    item["seq"],
                    item["stamp"],
                    item["jpg"],
                    result,
                )

    g["camera_loop"] = camera_loop
    g["detect_loop"] = detect_loop

    def status():
        return jsonify(
            ok=True,
            recorder=recorder.status(),
            baseline=model_info(),
        )

    def start():
        with g["control_lock"]:
            motion = (
                app.view_functions["motion_status"]()
                .get_json()
                .get("run")
            )

            if motion and motion["state"] == "running":
                return jsonify(
                    ok=False,
                    msg="Finish motion trial first",
                ), 409

            if (
                state["testing"]
                or state.get("motion_test_pending")
                or g["imu"].is_calibrating()
            ):
                return jsonify(
                    ok=False,
                    msg="Finish tests/calibration first",
                ), 409

            if (
                any(state["applied_drive"])
                or state["l"]
                or state["r"]
            ):
                return jsonify(
                    ok=False,
                    msg="Stop robot first",
                ), 409

            with g["cond"]:
                camera_age = (
                    time.monotonic()
                    - g["frame"]["stamp"]
                )

            if not state["cam"] or camera_age > 0.5:
                return jsonify(
                    ok=False,
                    msg="Fresh camera frames required",
                ), 409

            baseline = model_info()

            if (
                state["detect"]
                and not baseline["ready"]
            ):
                return jsonify(
                    ok=False,
                    msg="Wait for baseline loading/warm-up",
                ), 409

            metadata = dict(
                baseline=baseline,
                baseline_enabled=state["detect"],
                settings=copy.deepcopy(cfg),
                effective_imgsz=320,
                classes=["person"],
                nms_iou=0.7,
                max_det=100,
                timestamp_kind=(
                    "host monotonic camera capture completion"
                ),
            )

            try:
                identity = recorder.start(
                    request.get_json(force=True),
                    metadata,
                )

                return jsonify(
                    ok=True,
                    id=identity,
                )

            except (ValueError, TypeError, OSError) as error:
                return jsonify(
                    ok=False,
                    msg=str(error),
                ), 409

    def stop():
        recorder.stop()
        return jsonify(ok=True)

    def download(identity, filename):
        if (
            not re.fullmatch(
                r"[A-Za-z0-9_-]{1,100}",
                identity,
            )
            or filename not in (
                "report.json",
                "report.md",
                "bundle.zip",
                "raw.avi",
            )
        ):
            return jsonify(
                ok=False,
                msg="Unknown file",
            ), 404

        path = recorder.root / identity / filename

        if not path.is_file():
            return jsonify(
                ok=False,
                msg="File not ready",
            ), 404

        return send_file(
            path,
            as_attachment=True,
            download_name=identity + "_" + filename,
        )

    app.add_url_rule(
        "/pilot",
        "pilot_page",
        lambda: send_file(
            root / "dashboard" / "pilot.html"
        ),
    )
    app.add_url_rule(
        "/api/pilot/status",
        "pilot_status",
        status,
    )
    app.add_url_rule(
        "/api/pilot/start",
        "pilot_start",
        start,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/pilot/stop",
        "pilot_stop",
        stop,
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/pilot/file/<identity>/<filename>",
        "pilot_file",
        download,
    )

    def export_start():
        try:
            return jsonify(ok=True, export=recorder.export_all(
                "yolo_dataset" if request.args.get("purpose") == "yolo_dataset" else None))
        except (ValueError, TypeError, OSError) as error:
            return jsonify(ok=False, msg=str(error)), 409

    def export_download(identity):
        if not re.fullmatch(r'[a-f0-9]{32}', identity):
            return jsonify(ok=False, msg='Unknown archive'), 404
        path = recorder.root / '_exports' / (identity + '.zip')
        if not path.is_file():
            return jsonify(ok=False, msg='Archive not ready'), 404
        return send_file(path, as_attachment=True,
                         download_name='see_pilot_all_' + identity[:8] + '.zip')

    app.add_url_rule('/api/pilot/export-all', 'pilot_export_all', export_start, methods=['POST'])
    app.add_url_rule('/api/pilot/export-all/<identity>', 'pilot_export_download', export_download)


    def detection():
        if recorder.busy():
            return jsonify(
                ok=False,
                msg="Finish recording first",
            ), 409

        on = bool(
            request.get_json(force=True).get(
                "on",
                False,
            )
        )

        if on and not model_info()["ready"]:
            return jsonify(
                ok=False,
                msg="Baseline not ready; see pilot status",
            ), 409

        state["detect"] = on
        state["det_state"] = (
            "baseline running"
            if on
            else "ready / off"
        )

        return jsonify(ok=True)

    app.view_functions["api_detect"] = detection

    # Keep stationary recording and its settings consistent.
    for name in (
        "api_drive",
        "api_test",
        "api_settings_set",
        "api_safety",
        "api_detect",
        "api_imu_calibrate",
        "api_imu_zero",
        "motion_start",
        "api_talk",
    ):
        original = app.view_functions[name]

        def guarded(
            *args,
            _name=name,
            _original=original,
            **kwargs,
        ):
            with g["control_lock"]:
                if recorder.busy():
                    if (
                        _name != "api_drive"
                        or any(
                            float(
                                request.get_json(
                                    force=True
                                ).get(k, 0)
                            )
                            for k in ("l", "r")
                        )
                    ):
                        return jsonify(
                            ok=False,
                            msg=(
                                "Pilot recording: stop recording "
                                "before this action"
                            ),
                        ), 409

                if _name != "api_test":
                    return _original(
                        *args,
                        **kwargs,
                    )

                state["motion_test_pending"] = (
                    state.get(
                        "motion_test_pending",
                        0,
                    ) + 1
                )

            # Legacy tests may block; release motor control lock.
            try:
                return _original(
                    *args,
                    **kwargs,
                )
            finally:
                with g["control_lock"]:
                    state["motion_test_pending"] -= 1

        app.view_functions[name] = guarded

    original_cleanup = g["cleanup"]

    def cleanup():
        try:
            original_cleanup()
        finally:
            recorder.stop("server shutdown")

        with recorder.lock:
            thread = (
                recorder.run["thread"]
                if recorder.run
                else None
            )

        if thread:
            thread.join(timeout=15)

    g["cleanup"] = cleanup