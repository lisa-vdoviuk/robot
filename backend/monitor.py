"""Live HEAR; shares microphone ownership with the existing recorder."""
import atexit
import os
import select
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import jsonify, request, send_file
from vad_engine import Detector


def install(g, mic_lock, playback_hint):
    app = g["app"]
    lock = threading.RLock()
    stop = threading.Event()
    state = dict(
        mode="off",
        error="",
        speech=False,
        score=None,
        playback=False,
        updated=0.0,
        last_voice=None,
        events=0,
    )
    thread = None

    def update(**values):
        with lock:
            state.update(values)

    def worker():
        proc = None
        failed = False
        try:
            detector = Detector()
            device = g["find_mic"]()
            if not device:
                raise RuntimeError("No USB microphone found")

            with tempfile.TemporaryFile() as errors:
                proc = subprocess.Popen(
                    [
                        "arecord", "-q", "-D", device,
                        "-t", "raw", "-f", "S16_LE",
                        "-r", "48000", "-c", "1",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=errors,
                    bufsize=0,
                )
                update(mode="listening")
                last_data = time.monotonic()
                pending = b""
                in_event = False

                while not stop.is_set():
                    ready, _, _ = select.select(
                        [proc.stdout], [], [], 0.2
                    )

                    if not ready:
                        if time.monotonic() - last_data > 3:
                            raise RuntimeError(
                                "Microphone stalled for 3 seconds"
                            )
                        continue

                    raw = os.read(proc.stdout.fileno(), 2880)
                    if not raw:
                        errors.seek(0)
                        raise RuntimeError(
                            "Capture ended: "
                            + errors.read(2000).decode(errors="replace")
                        )

                    last_data = time.monotonic()
                    pending += raw
                    size = len(pending) // 2 * 2
                    if not size:
                        continue

                    rows = detector.feed(
                        pending[:size], playback_hint()
                    )
                    pending = pending[size:]

                    for row in rows:
                        # Keep the playback flag until this event closes.
                        own = row["playback_possible"] or bool(
                            detector.event
                            and detector.event["playback_possible"]
                        )
                        active = row["speech_active"]

                        with lock:
                            if active and not in_event:
                                state["events"] += 1
                            if active and not own:
                                state["last_voice"] = (
                                    datetime.now(timezone.utc).isoformat()
                                )

                        in_event = active
                        update(
                            speech=active,
                            playback=own,
                            score=row["speech_score"],
                            updated=time.monotonic(),
                        )

                    # Do not accumulate history during continuous monitoring.
                    detector.events.clear()
                    detector.times.clear()

        except Exception as exc:
            failed = True
            update(error=str(exc))

        finally:
            if proc is not None:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=2)
                proc.stdout.close()

            update(
                mode="error" if failed and not stop.is_set() else "off",
                speech=False,
                playback=False,
                score=None,
                updated=0.0,
            )
            mic_lock.release()

    @app.post("/api/monitor")
    def control():
        nonlocal thread
        body = request.get_json(silent=True) or {}

        if type(body.get("enabled")) is not bool:
            return jsonify(
                error="enabled must be true or false"
            ), 400

        with lock:
            if not body["enabled"]:
                stop.set()
                if thread and thread.is_alive():
                    state.update(mode="stopping", speech=False)
                return jsonify(ok=True)

            if thread and thread.is_alive():
                if stop.is_set():
                    return jsonify(
                        error="Still stopping; retry shortly"
                    ), 409
                return jsonify(ok=True)

            if not mic_lock.acquire(False):
                return jsonify(
                    error=(
                        "Microphone busy: finish the HEAR "
                        "recording or audio test"
                    )
                ), 409

            stop.clear()
            state.update(
                mode="starting",
                error="",
                speech=False,
                score=None,
                playback=False,
                updated=0.0,
                events=0,
                last_voice=None,
            )

            thread = threading.Thread(
                target=worker, daemon=True
            )
            try:
                thread.start()
            except Exception:
                state["mode"] = "error"
                mic_lock.release()
                raise

        return jsonify(ok=True)

    @app.get("/api/monitor")
    def status():
        with lock:
            data = dict(state)

        now = time.monotonic()
        data["fresh"] = (
            data["mode"] == "listening"
            and now - data.pop("updated") < 2
        )
        data["speech"] = data["speech"] and data["fresh"]

        s = g["S"]
        stamp = s.get("see_result_at", 0)
        see_fresh = bool(
            s.get("cam")
            and s.get("detect")
            and stamp
            and now - stamp < 2
        )
        person = see_fresh and "person" in s.get("alert", [])
        voice = data["speech"] and not data["playback"]

        data.update(
            see_fresh=see_fresh,
            person=person,
            signal=(
                "both" if person and voice
                else "person" if person
                else "voice" if voice
                else "none"
            ),
        )
        return jsonify(data)

    @app.get("/monitor.js")
    def script():
        return send_file(
            Path(g["HERE"]) / "dashboard" / "monitor.js"
        )

    @app.get("/mission.css")
    def style():
        return send_file(
            Path(g["HERE"]) / "dashboard" / "mission.css"
        )

    def cleanup():
        stop.set()
        if thread:
            thread.join(timeout=4)

    atexit.register(cleanup)