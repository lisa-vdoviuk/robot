"""Stationary gyro bias calibration and short-term, level-floor relative yaw."""
import statistics
import time
from imu import MPU6050 as BaseMPU6050


class MPU6050(BaseMPU6050):
    def __init__(self, bus_number=1):
        super().__init__(bus_number)
        self.cal_start = None
        self.samples = []
        self.previous = None
        self.yaw_start = None
        self.state.update(
            calibration="needed",
            calibration_error="",
            progress=0,
            bias_dps=None,
            gyro_std_dps=None,
            sample_count=0,
            calibration_temp_c=None,
            corrected_dps=None,
            yaw_deg=None,
            yaw_valid=False,
        )

    def is_calibrating(self):
        with self.lock:
            return self.cal_start is not None

    def start_calibration(self):
        with self.lock:
            now = time.monotonic()
            ts = self.state["ts_mono"]

            if self.cal_start is not None:
                return False, "Calibration already running"

            if not self.state["valid"] or ts is None or now - ts > 0.2:
                return False, "No fresh IMU data"

            self.cal_start = now
            self.samples = []
            self.previous = None
            self.yaw_start = None

            self.state.update(
                calibration="running",
                calibration_error="",
                progress=0,
                bias_dps=None,
                gyro_std_dps=None,
                sample_count=0,
                calibration_temp_c=None,
                corrected_dps=None,
                yaw_deg=None,
                yaw_valid=False,
            )
            return True, "Keep robot still and level for 6 seconds"

    def fail_calibration(self, message):
        # Caller holds self.lock.
        self.cal_start = None
        self.samples = []
        self.state.update(
            calibration="failed",
            calibration_error=message,
            yaw_valid=False,
            yaw_deg=None,
        )

    def update_estimate(self, now):
        # Called with self.lock after a fresh hardware sample.
        s = self.state
        a, g = s["accel_g"], s["gyro_dps"]
        dt = None if self.previous is None else now - self.previous[0]
        saturated = any(abs(v) >= 249 for v in g)

        if self.cal_start is not None:
            elapsed = now - self.cal_start
            s["progress"] = min(100, round(elapsed / 6 * 100))

            # First second lets the robot settle after stopping its motors.
            if elapsed >= 1:
                if dt is None or not 0 < dt <= 0.2:
                    self.fail_calibration("Sample gap: retry calibration")

                elif not 0.85 <= s["accel_norm_g"] <= 1.15 or a[2] < 0.9:
                    self.fail_calibration(
                        "Keep board flat and robot stationary"
                    )

                elif max(abs(v) for v in g) > 5:
                    self.fail_calibration(
                        "Rotation or large gyro offset: inspect raw data"
                    )

                else:
                    self.samples.append((a[:], g[:], s["temp_c"]))
                    s["sample_count"] = len(self.samples)

                    if elapsed >= 6:
                        if len(self.samples) < 150:
                            self.fail_calibration(
                                "Too few samples: retry with less CPU load"
                            )
                        else:
                            gs = [
                                statistics.pstdev(
                                    x[1][i] for x in self.samples
                                )
                                for i in range(3)
                            ]
                            ast = [
                                statistics.pstdev(
                                    x[0][i] for x in self.samples
                                )
                                for i in range(3)
                            ]

                            if max(gs) > 0.35 or max(ast) > 0.02:
                                self.fail_calibration(
                                    "Movement/vibration detected: retry"
                                )
                            else:
                                s.update(
                                    bias_dps=[
                                        statistics.mean(
                                            x[1][i] for x in self.samples
                                        )
                                        for i in range(3)
                                    ],
                                    gyro_std_dps=gs,
                                    calibration_temp_c=statistics.mean(
                                        x[2] for x in self.samples
                                    ),
                                    calibration="ready",
                                    progress=100,
                                    yaw_deg=0.0,
                                    yaw_valid=True,
                                )
                                self.yaw_start = now
                                self.cal_start = None
                                self.samples = []
                                dt = None

        if s["bias_dps"] is not None:
            corrected = [
                v - b for v, b in zip(g, s["bias_dps"])
            ]
            s["corrected_dps"] = corrected

            if saturated or (dt is not None and not 0 < dt <= 0.2):
                s.update(
                    yaw_valid=False,
                    yaw_deg=None,
                    calibration_error=(
                        "Angle lost: sample gap or gyro range exceeded; "
                        "reset angle"
                    ),
                )

            if s["yaw_valid"] and dt is not None:
                old_rate = self.previous[1] - s["bias_dps"][2]
                s["yaw_deg"] += (old_rate + corrected[2]) * 0.5 * dt

        self.previous = (now, g[2])

    def zero_yaw(self):
        with self.lock:
            now = time.monotonic()
            s = self.state

            if (
                self.cal_start is not None
                or s["bias_dps"] is None
                or not s["valid"]
                or s["ts_mono"] is None
                or now - s["ts_mono"] > 0.2
                or any(abs(v) >= 249 for v in s["gyro_dps"])
            ):
                return False, "Fresh data and successful calibration required"

            self.previous = None
            self.yaw_start = now
            s.update(
                yaw_deg=0.0,
                yaw_valid=True,
                calibration_error="",
            )
            return True, "Relative angle reset to zero"

    def publish(self, **values):
        with self.lock:
            self.state.update(values)

            if values.get("valid") is False:
                if self.cal_start is not None:
                    self.fail_calibration(
                        "Sensor disconnected during calibration"
                    )

                self.state.update(
                    bias_dps=None,
                    corrected_dps=None,
                    yaw_valid=False,
                    yaw_deg=None,
                )
                if self.state["calibration"] != "failed":
                    self.state["calibration"] = "needed"

                self.previous = None

    def snapshot(self):
        # Base snapshot also masks stale accelerometer/gyro readings.
        result = super().snapshot()

        with self.lock:
            now = time.monotonic()

            if self.cal_start is not None and now - self.cal_start > 8:
                self.fail_calibration(
                    "Calibration timed out: no continuous data"
                )
                result.update(
                    calibration="failed",
                    calibration_error=self.state["calibration_error"],
                )

            result["yaw_elapsed_s"] = (
                None if self.yaw_start is None
                else round(now - self.yaw_start, 1)
            )

        if not result["valid"]:
            result.update(
                corrected_dps=None,
                yaw_deg=None,
                yaw_valid=False,
            )

        return result