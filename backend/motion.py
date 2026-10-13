"""Short supervised motion trials using the existing motor control loop."""
import copy
import json
import math
import os
import statistics
import threading
import time
import uuid

from flask import jsonify, request, send_file


class Trials:
    def __init__(self):
        self.run = None
        self.deadline = 0

    def active(self):
        return (
            self.run is not None
            and self.run["state"] == "running"
        )

    def begin(self, data, cfg, yaw, now):
        if self.run and not self.run.get("saved"):
            raise ValueError(
                "Save or discard the previous trial first"
            )

        power = float(data["power"])
        duration = float(data["duration"])

        if not math.isfinite(power) or not 0.2 <= power <= 0.6:
            raise ValueError("Power must be 0.20 to 0.60")

        if not math.isfinite(duration) or not 0.5 <= duration <= 3:
            raise ValueError("Duration must be 0.5 to 3 seconds")

        direction = data["direction"]
        if direction not in ("forward", "reverse"):
            raise ValueError("Invalid direction")

        surface = str(data.get("surface", "")).strip()[:100]
        if not surface:
            raise ValueError("Enter the floor surface")

        self.run = dict(
            id=uuid.uuid4().hex,
            date=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            state="running",
            reason="",
            saved=False,
            power=power,
            duration_s=duration,
            direction=direction,
            surface=surface,
            battery=str(data.get("battery", "unknown"))[:80],
            cfg={
                k: cfg[k]
                for k in (
                    "max_speed",
                    "trim_l",
                    "trim_r",
                    "swap",
                    "stop_cm",
                )
            },
            start=None,
            elapsed_s=0,
            yaw_start=yaw,
            yaw_delta_deg=0,
            max_yaw_deg=0,
            applied=None,
        )

        self.deadline = now + 0.8

    def finish(self, state, reason, now):
        if self.active():
            run = self.run
            run.update(
                state=state,
                reason=reason,
                elapsed_s=(
                    0 if run["start"] is None
                    else now - run["start"]
                ),
            )

    def command(self, now, imu_data):
        if not self.active():
            return 0

        run = self.run

        if (
            not imu_data.get("valid")
            or not imu_data.get("yaw_valid")
        ):
            self.finish("aborted", "IMU data lost", now)
            return 0

        run["yaw_delta_deg"] = (
            imu_data["yaw_deg"] - run["yaw_start"]
        )
        run["max_yaw_deg"] = max(
            run["max_yaw_deg"],
            abs(run["yaw_delta_deg"]),
        )

        if now > self.deadline:
            self.finish(
                "aborted",
                "Browser heartbeat lost",
                now,
            )

        elif (
            run["start"] is not None
            and now - run["start"] >= run["duration_s"]
        ):
            self.finish(
                "complete",
                "Timed command ended",
                now,
            )

        if not self.active():
            return 0

        sign = 1 if run["direction"] == "forward" else -1
        return run["power"] * sign


def install(g):
    """Call once before starting the server workers."""
    app = g["app"]
    state = g["S"]
    cfg = g["CFG"]
    imu = g["imu"]
    motors = g["motors"]
    drive = g["drive"]

    lock = g["control_lock"] = threading.RLock()
    trials = Trials()
    file_lock = threading.Lock()

    output_path = os.path.join(
        g["HERE"], "motion_calibration.json"
    )

    state["sonar_ts"] = [None, None]

    def stop(state_name="aborted", reason="Manual STOP"):
        trials.finish(
            state_name,
            reason,
            time.monotonic(),
        )

        state.update(
            l=0.0,
            r=0.0,
            t=0.0,
            applied_drive=[0.0, 0.0],
        )

        for motor in motors:
            drive(motor, 0)

    def sonar_loop():
        history = [[], []]

        while True:
            for i, sensor in enumerate(g["sonars"]):
                if sensor is not None:
                    try:
                        distance = sensor.distance * 100

                        if (
                            not math.isfinite(distance)
                            or distance < 0
                        ):
                            raise ValueError(
                                "Invalid sonar distance"
                            )

                        history[i] = (
                            history[i] + [distance]
                        )[-3:]

                        with lock:
                            state["dist"][i] = round(
                                statistics.median(history[i]),
                                1,
                            )
                            state["sonar_ts"][i] = (
                                time.monotonic()
                            )

                    except Exception:
                        with lock:
                            state["dist"][i] = None
                            state["sonar_ts"][i] = None

                    time.sleep(0.05)

            time.sleep(0.05)

    def control():
        while True:
            with lock:
                now = time.monotonic()
                was_trial = trials.active()

                if was_trial:
                    value = trials.command(
                        now, imu.snapshot()
                    )
                    left = right = value

                    if trials.active():
                        index = (
                            g["FRONT_SONAR"]
                            if value > 0
                            else g["REAR_SONAR"]
                        )
                        timestamp = state["sonar_ts"][index]

                        if (
                            timestamp is None
                            or now - timestamp > 0.5
                            or state["dist"][index] is None
                        ):
                            trials.finish(
                                "aborted",
                                "Sonar reads unavailable/stale",
                                now,
                            )
                            left = right = 0

                else:
                    left = state["l"]
                    right = state["r"]

                    if (
                        time.time() - state["t"]
                        > cfg["watchdog"]
                    ):
                        left = right = 0

                if imu.is_calibrating():
                    left = right = 0

                if not state["testing"]:
                    cap = cfg["max_speed"]
                    a, b = (
                        (right, left)
                        if cfg["swap"]
                        else (left, right)
                    )

                    a = max(
                        -1.0,
                        min(
                            1.0,
                            max(-cap, min(cap, a))
                            * cfg["trim_l"],
                        ),
                    )
                    b = max(
                        -1.0,
                        min(
                            1.0,
                            max(-cap, min(cap, b))
                            * cfg["trim_r"],
                        ),
                    )

                    front, rear = g["sonar_ranges"]()

                    blocked = g["obstacle_blocks"](
                        a,
                        b,
                        front,
                        rear,
                        cfg["stop_cm"],
                        state["safety"] or was_trial,
                    )

                    state.update(
                        blocked_by=blocked,
                        blocked=bool(blocked),
                    )

                    if blocked:
                        a = b = 0
                        trials.finish(
                            "aborted",
                            "Sonar obstacle STOP",
                            now,
                        )

                    state["applied_drive"] = [a, b]

                    try:
                        drive(motors[0], a)
                        drive(motors[1], b)

                    except Exception as error:
                        stop(
                            reason=(
                                "Motor output error: "
                                + str(error)[:100]
                            )
                        )

                    if trials.active():
                        run = trials.run

                        if run["start"] is None:
                            run["start"] = time.monotonic()

                        run["elapsed_s"] = (
                            time.monotonic() - run["start"]
                        )
                        run["applied"] = [a, b]

            time.sleep(0.02)

    g["control"] = control
    g["sonar_loop"] = sonar_loop

    # Existing routes cannot change configuration or start
    # another test during an active motion trial.
    guarded_routes = (
        "api_drive",
        "api_test",
        "api_settings_set",
        "api_safety",
        "api_imu_calibrate",
        "api_imu_zero",
    )

    for name in guarded_routes:
        original = app.view_functions[name]

        def guarded(*args, _original=original, **kwargs):
            with lock:
                if trials.active():
                    return jsonify(
                        ok=False,
                        msg="Motion trial running; use STOP",
                    ), 409

                if _original.__name__ == "api_test":
                    state["motion_test_pending"] = (
                        state.get("motion_test_pending", 0)
                        + 1
                    )
                else:
                    return _original(*args, **kwargs)

            # Legacy hardware tests may block. Do not hold
            # the motor-control lock while they execute.
            try:
                return _original(*args, **kwargs)
            finally:
                with lock:
                    state["motion_test_pending"] -= 1

        app.view_functions[name] = guarded

    original_stop = app.view_functions["api_stop"]

    def api_stop():
        with lock:
            stop()
            return original_stop()

    app.view_functions["api_stop"] = api_stop

    def status():
        with lock:
            return jsonify(
                ok=True,
                run=copy.deepcopy(trials.run),
            )

    app.add_url_rule(
        "/api/motion/status",
        "motion_status",
        status,
    )

    def start():
        with lock:
            try:
                imu_data = imu.snapshot()

                if (
                    state["testing"]
                    or state.get("motion_test_pending", 0)
                    or imu.is_calibrating()
                    or trials.active()
                ):
                    raise ValueError(
                        "Another test/calibration is running"
                    )

                if (
                    not state["safety"]
                    or any(m is None for m in motors)
                ):
                    raise ValueError(
                        "Enable sonar safety; both motors "
                        "must be available"
                    )

                if (
                    any(
                        abs(v) > 0.001
                        for v in state["applied_drive"]
                    )
                    or state["l"]
                    or state["r"]
                ):
                    raise ValueError(
                        "Stop the robot before starting"
                    )

                if (
                    not imu_data.get("valid")
                    or not imu_data.get("yaw_valid")
                ):
                    raise ValueError(
                        "Calibrate IMU and reset its angle first"
                    )

                trials.begin(
                    request.get_json(force=True),
                    cfg,
                    imu_data["yaw_deg"],
                    time.monotonic(),
                )

                state.update(l=0.0, r=0.0, t=0.0)

                return jsonify(
                    ok=True,
                    run=copy.deepcopy(trials.run),
                )

            except (ValueError, KeyError, TypeError) as error:
                return jsonify(
                    ok=False,
                    msg=str(error),
                ), 409

    app.add_url_rule(
        "/api/motion/start",
        "motion_start",
        start,
        methods=["POST"],
    )

    def heartbeat():
        with lock:
            data = request.get_json(force=True)

            if (
                not trials.active()
                or data.get("id") != trials.run["id"]
            ):
                return jsonify(ok=False), 409

            if time.monotonic() > trials.deadline:
                stop(reason="Browser heartbeat lost")
                return jsonify(
                    ok=False,
                    msg="Heartbeat expired",
                ), 409

            trials.deadline = time.monotonic() + 0.8
            return jsonify(ok=True)

    app.add_url_rule(
        "/api/motion/heartbeat",
        "motion_heartbeat",
        heartbeat,
        methods=["POST"],
    )

    def discard():
        with lock:
            if trials.active():
                return jsonify(
                    ok=False,
                    msg="Stop first",
                ), 409

            trials.run = None
            return jsonify(ok=True)

    app.add_url_rule(
        "/api/motion/discard",
        "motion_discard",
        discard,
        methods=["POST"],
    )

    def save():
        try:
            data = request.get_json(force=True)
            distance = float(data["distance_cm"])

            if (
                not math.isfinite(distance)
                or not 0 < distance <= 1000
            ):
                raise ValueError(
                    "Enter measured distance: "
                    ">0 and <=1000 cm"
                )

            with file_lock:
                with lock:
                    run = copy.deepcopy(trials.run)

                    if (
                        not run
                        or run["state"] == "running"
                        or run["id"] != data.get("id")
                    ):
                        raise ValueError(
                            "Select the completed trial"
                        )

                run.update(
                    distance_cm=distance,
                    notes=str(data.get("notes", ""))[:300],
                )

                run["usable"] = (
                    run["state"] == "complete"
                    and run["elapsed_s"] > 0
                    and run["max_yaw_deg"] <= 5
                )

                run["speed_cm_s"] = (
                    distance / run["elapsed_s"]
                    if run["elapsed_s"] > 0
                    else None
                )
                run["saved"] = True

                document = {
                    "schema": 1,
                    "trials": [],
                }

                if os.path.exists(output_path):
                    with open(
                        output_path, encoding="utf-8"
                    ) as file:
                        document = json.load(file)

                document["trials"] = [
                    item
                    for item in document["trials"]
                    if item["id"] != run["id"]
                ] + [run]

                condition_keys = (
                    "direction",
                    "power",
                    "duration_s",
                    "surface",
                    "battery",
                    "cfg",
                )

                group = [
                    item["speed_cm_s"]
                    for item in document["trials"]
                    if item["usable"]
                    and all(
                        item[key] == run[key]
                        for key in condition_keys
                    )
                ]

                # This model applies only to matching conditions.
                document["latest_model"] = dict(
                    conditions={
                        key: run[key]
                        for key in condition_keys
                    },
                    count=len(group),
                    ready=len(group) >= 3,
                    mean_cm_s=(
                        statistics.mean(group)
                        if group else None
                    ),
                    std_cm_s=(
                        statistics.pstdev(group)
                        if group else None
                    ),
                )

                with open(
                    output_path + ".tmp",
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(
                        document,
                        file,
                        indent=2,
                        ensure_ascii=False,
                        allow_nan=False,
                    )

                os.replace(
                    output_path + ".tmp",
                    output_path,
                )

                with lock:
                    if (
                        trials.run
                        and trials.run["id"] == run["id"]
                    ):
                        trials.run.update(run)

                return jsonify(
                    ok=True,
                    model=document["latest_model"],
                    run=run,
                )

        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
        ) as error:
            return jsonify(
                ok=False,
                msg=str(error),
            ), 400

    app.add_url_rule(
        "/api/motion/save",
        "motion_save",
        save,
        methods=["POST"],
    )

    def export():
        if not os.path.isfile(output_path):
            return jsonify(
                ok=False,
                msg="Save a measurement first",
            ), 404

        return send_file(
            output_path,
            as_attachment=True,
            download_name="motion_calibration.json",
        )

    app.add_url_rule(
        "/api/motion/export",
        "motion_export",
        export,
    )

    app.add_url_rule(
        "/motion",
        "motion_page",
        lambda: send_file(
            os.path.join(
                g["HERE"],
                "dashboard",
                "motion.html",
            )
        ),
    )