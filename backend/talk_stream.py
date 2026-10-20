"""Ordered PCM transport with bounded buffering and explicit failures."""
import atexit
import queue
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from flask import jsonify, request, send_file


def install(g):
    app, state = g["app"], g["S"]
    lock = threading.RLock()
    audio = queue.Queue(maxsize=12)
    session = dict(id=None, seq=0, closing=False, last=0, error="")
    shutdown = threading.Event()

    @app.post("/api/talk/start")
    def start():
        with lock:
            if session["id"]:
                return jsonify(error="Previous Talk session is still closing"), 409
            session.update(id=uuid.uuid4().hex, seq=0, closing=False,
                           last=time.monotonic(), error="")
            state["talk_err"] = ""
            return jsonify(id=session["id"])

    def receive():
        if request.content_length is None or request.content_length > 6400:
            return jsonify(error="Audio block size missing or too large"), 413
        raw = request.get_data(cache=False)
        sid = request.headers.get("X-Talk-Session")
        try:
            seq = int(request.headers.get("X-Talk-Sequence", "-1"))
        except ValueError:
            return jsonify(error="Invalid sequence"), 400
        if not raw or len(raw) > 6400 or len(raw) % 2:
            return jsonify(error="Expected at most 200 ms of PCM16 / 16 kHz"), 400
        with lock:
            if sid != session["id"] or not sid or session["closing"]:
                return jsonify(error="Talk session expired"), 409
            if seq < session["seq"]:
                return "", 204  # Retry of an already accepted block.
            if seq != session["seq"]:
                return jsonify(error="Out-of-order audio"), 409
            try:
                audio.put_nowait(raw)
            except queue.Full:
                return jsonify(error="Speaker buffer full"), 429
            session["seq"] += 1
            session["last"] = time.monotonic()
            state["talk_t"] = time.time()
        return "", 204

    def finish():
        with lock:
            if request.headers.get("X-Talk-Session") != session["id"]:
                return jsonify(error="Talk session expired"), 409
            session["closing"] = True
        return "", 204

    app.view_functions["api_talk"] = receive
    app.view_functions["api_talk_stop"] = finish

    @app.get("/talk.js")
    def browser():
        return send_file(Path(g["HERE"]) / "dashboard" / "talk.js")

    @app.get("/talk-worklet.js")
    def worklet():
        return send_file(Path(g["HERE"]) / "dashboard" / "talk-worklet.js")

    def worker():
        while not shutdown.is_set():
            with lock:
                sid = session["id"]
            if not sid:
                shutdown.wait(.05)
                continue
            player = None
            try:
                # Prebuffer two 200 ms chunks, or play a short completed phrase.
                while not shutdown.is_set():
                    with lock:
                        closing = session["closing"]
                        idle = time.monotonic() - session["last"]
                    if audio.qsize() >= 2 or closing:
                        break
                    if idle > 3:
                        raise RuntimeError("Talk timed out before audio arrived")
                    shutdown.wait(.02)
                with tempfile.TemporaryFile() as errors:
                    while not shutdown.is_set():
                        try:
                            chunk = audio.get(timeout=.1)
                        except queue.Empty:
                            with lock:
                                if session["closing"]:
                                    break
                                idle = time.monotonic() - session["last"]
                            if idle > 3:
                                raise RuntimeError("Talk connection timed out")
                            continue
                        if player is None:
                            player = subprocess.Popen(
                                ["aplay", "-q", "-D", g["SPK"], "-t", "raw",
                                 "-f", "S16_LE", "-r", "16000", "-c", "1",
                                 "-B", "400000"],
                                stdin=subprocess.PIPE, stderr=errors)
                        # A watchdog also bounds a hung ALSA write.
                        timer = threading.Timer(2, player.kill)
                        timer.daemon = True
                        timer.start()
                        try:
                            player.stdin.write(chunk)
                            player.stdin.flush()
                        finally:
                            timer.cancel()
                        state["talk_t"] = time.time()
                    if player:
                        player.stdin.close()
                        player.wait(timeout=3)
                        if player.returncode:
                            errors.seek(0)
                            raise RuntimeError(errors.read(2000).decode(errors="replace"))
            except Exception as exc:
                state["talk_err"] = str(exc)
            finally:
                if player:
                    if player.poll() is None:
                        player.kill()
                        player.wait()
                    if not player.stdin.closed:
                        player.stdin.close()
                with lock:
                    while not audio.empty():
                        audio.get_nowait()
                    session.update(id=None, closing=False)
                    state["talk_t"] = 0

    g["talk_worker"] = worker
    atexit.register(shutdown.set)
