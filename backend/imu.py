"""
MPU6050 telemetry, stationary gyro calibration and relative Z rotation.

Mount the board flat, components facing up.
Relative Z rotation assumes motion on a level floor.
Calibration is held in memory and must be repeated after restart.
"""
import math
import statistics
import struct
import threading
import time


def decode_sample(data):
    if len(data) != 14:
        raise ValueError("MPU6050: expected 14 bytes")

    ax, ay, az, temperature, gx, gy, gz = struct.unpack(
        ">7h", bytes(data)
    )

    accel = [v / 16384.0 for v in (ax, ay, az)]
    gyro = [v / 131.0 for v in (gx, gy, gz)]
    norm = math.sqrt(sum(v * v for v in accel))

    return dict(
        accel_g=accel,
        gyro_dps=gyro,
        temp_c=round(temperature / 340.0 + 36.53, 2),
        accel_norm_g=round(norm, 4),
    )


class MPU6050:
    def __init__(self, bus_number=1):
        self.bus_number = bus_number
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

        self.cal_start = None
        self.samples = []
        self.previous = None
        self.yaw_start = None

        self.state = dict(
            available=False,
            valid=False,
            status="starting",
            error="",
            bus=bus_number,
            address=None,
            who_am_i=None,
            seq=0,
            ts_mono=None,
            accel_g=None,
            gyro_dps=None,
            temp_c=None,
            accel_norm_g=None,
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
        with self.lock:
            now = time.monotonic()

            if self.cal_start is not None and now - self.cal_start > 8:
                self.fail_calibration(
                    "Calibration timed out: no continuous data"
                )

            result = dict(self.state)
            result["yaw_elapsed_s"] = (
                None if self.yaw_start is None
                else round(now - self.yaw_start, 1)
            )

        ts = result["ts_mono"]
        age = None if ts is None else max(0, now - ts)

        result["age_ms"] = (
            None if age is None else round(age * 1000)
        )
        result["valid"] = bool(
            result["valid"]
            and age is not None
            and age <= 0.5
        )

        if not result["valid"]:
            for key in (
                "accel_g",
                "gyro_dps",
                "temp_c",
                "accel_norm_g",
                "corrected_dps",
                "yaw_deg",
            ):
                result[key] = None

            result["yaw_valid"] = False

            if result["status"] == "running":
                result["status"] = "stale"

        return result

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(
                target=self.run,
                daemon=True,
                name="mpu6050",
            )
            self.thread.start()

    def close(self):
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=1)

    def connect(self, bus):
        errors = []

        for address in (0x68, 0x69):
            try:
                who = bus.read_byte_data(address, 0x75)

                if who != 0x68:
                    raise ValueError(
                        f"unexpected WHO_AM_I 0x{who:02x}"
                    )

            except (OSError, ValueError) as e:
                errors.append(f"0x{address:02x}: {e}")
                continue

            # PLL X gyro clock, all axes enabled, DLPF,
            # 50 Hz, +/-250 deg/s and +/-2 g.
            settings = (
                (0x6B, 0x01),
                (0x6C, 0x00),
                (0x1A, 0x03),
                (0x19, 19),
                (0x1B, 0x00),
                (0x1C, 0x00),
                (0x38, 0x01),
            )

            for register, value in settings:
                bus.write_byte_data(address, register, value)

            if self.stop_event.wait(0.1):
                return None

            for register, value in settings:
                if bus.read_byte_data(address, register) != value:
                    raise OSError(
                        "MPU6050 configuration mismatch at "
                        f"0x{register:02x}"
                    )

            self.publish(
                available=True,
                address=f"0x{address:02x}",
                who_am_i=f"0x{who:02x}",
                status="waiting for data",
                error="",
                valid=False,
            )
            return address

        raise OSError(
            "MPU6050 not found on I2C bus %s; %s"
            % (self.bus_number, "; ".join(errors))
        )

    def run(self):
        try:
            try:
                from smbus import SMBus
            except ImportError:
                from smbus2 import SMBus

        except ImportError:
            self.publish(
                status="dependency missing",
                error="Install python3-smbus (or smbus2 in your venv)",
            )
            return

        while not self.stop_event.is_set():
            bus = None

            try:
                bus = SMBus(self.bus_number)
                address = self.connect(bus)

                if address is None:
                    break

                last_data = time.monotonic()

                while not self.stop_event.is_set():
                    if bus.read_byte_data(address, 0x3A) & 1:
                        values = decode_sample(
                            bus.read_i2c_block_data(
                                address, 0x3B, 14
                            )
                        )
                        last_data = time.monotonic()

                        with self.lock:
                            self.state.update(values)
                            self.state.update(
                                valid=True,
                                available=True,
                                status="running",
                                error="",
                                ts_mono=last_data,
                                seq=self.state["seq"] + 1,
                            )
                            self.update_estimate(last_data)

                    elif time.monotonic() - last_data > 1:
                        raise OSError(
                            "MPU6050: no fresh data-ready signal"
                        )

                    self.stop_event.wait(0.02)

            except Exception as e:
                self.publish(
                    available=False,
                    valid=False,
                    status="unavailable",
                    error=str(e)[:180],
                )

            finally:
                if bus is not None:
                    try:
                        bus.close()
                    except Exception:
                        pass

            self.stop_event.wait(2)

    def is_calibrating(self):
        with self.lock:
            return self.cal_start is not None

    def start_calibration(self):
        with self.lock:
            now = time.monotonic()
            ts = self.state["ts_mono"]

            if self.cal_start is not None:
                return False, "Calibration already running"

            if (
                not self.state["valid"]
                or ts is None
                or now - ts > 0.2
            ):
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
        # Caller holds self.lock.
        s = self.state
        a = s["accel_g"]
        g = s["gyro_dps"]

        dt = (
            None if self.previous is None
            else now - self.previous[0]
        )
        saturated = any(abs(v) >= 249 for v in g)

        if self.cal_start is not None:
            elapsed = now - self.cal_start
            s["progress"] = min(
                100, round(elapsed / 6 * 100)
            )

            # One second to settle, followed by five seconds of samples.
            if elapsed >= 1:
                if dt is None or not 0 < dt <= 0.2:
                    self.fail_calibration(
                        "Sample gap: retry calibration"
                    )

                elif (
                    not 0.85 <= s["accel_norm_g"] <= 1.15
                    or a[2] < 0.9
                ):
                    self.fail_calibration(
                        "Keep board flat and robot stationary"
                    )

                elif max(abs(v) for v in g) > 5:
                    self.fail_calibration(
                        "Rotation or large gyro offset: inspect raw data"
                    )

                else:
                    self.samples.append(
                        (a[:], g[:], s["temp_c"])
                    )
                    s["sample_count"] = len(self.samples)

                    if elapsed >= 6:
                        if len(self.samples) < 150:
                            self.fail_calibration(
                                "Too few samples: retry with less CPU load"
                            )

                        else:
                            gyro_std = [
                                statistics.pstdev(
                                    sample[1][i]
                                    for sample in self.samples
                                )
                                for i in range(3)
                            ]
                            accel_std = [
                                statistics.pstdev(
                                    sample[0][i]
                                    for sample in self.samples
                                )
                                for i in range(3)
                            ]

                            if (
                                max(gyro_std) > 0.35
                                or max(accel_std) > 0.02
                            ):
                                self.fail_calibration(
                                    "Movement/vibration detected: retry"
                                )

                            else:
                                bias = [
                                    statistics.mean(
                                        sample[1][i]
                                        for sample in self.samples
                                    )
                                    for i in range(3)
                                ]
                                temperature = statistics.mean(
                                    sample[2]
                                    for sample in self.samples
                                )

                                s.update(
                                    bias_dps=bias,
                                    gyro_std_dps=gyro_std,
                                    calibration_temp_c=temperature,
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
                value - bias
                for value, bias in zip(g, s["bias_dps"])
            ]
            s["corrected_dps"] = corrected

            if (
                saturated
                or (dt is not None and not 0 < dt <= 0.2)
            ):
                s.update(
                    yaw_valid=False,
                    yaw_deg=None,
                    calibration_error=(
                        "Angle lost: sample gap or gyro range exceeded; "
                        "reset angle"
                    ),
                )

            if s["yaw_valid"] and dt is not None:
                old_rate = (
                    self.previous[1] - s["bias_dps"][2]
                )
                s["yaw_deg"] += (
                    (old_rate + corrected[2]) * 0.5 * dt
                )

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
                return (
                    False,
                    "Fresh data and successful calibration required",
                )

            self.previous = None
            self.yaw_start = now

            s.update(
                yaw_deg=0.0,
                yaw_valid=True,
                calibration_error="",
            )
            return True, "Relative angle reset to zero"

    def test(self):
        first = self.snapshot()
        time.sleep(0.3)
        current = self.snapshot()

        if not current["valid"]:
            return (
                False,
                current["error"]
                or "No fresh MPU6050 samples; check I2C, wiring and power",
            )

        if current["seq"] <= first["seq"]:
            return False, "MPU6050 samples are not updating"

        norm = current["accel_norm_g"]

        if not 0.5 <= norm <= 1.5:
            return (
                False,
                f"Fresh data, but |a|={norm:.3f} g; "
                "keep robot still and repeat",
            )

        return True, (
            f"I2C {current['bus']} / {current['address']}, "
            f"WHO_AM_I={current['who_am_i']}; "
            f"fresh samples, |a|={norm:.3f} g; "
            f"gyro={current['gyro_dps']} deg/s. "
            "Communication/basic gravity check only; "
            "axis response and calibration remain manual."
        )


if __name__ == "__main__":
    import json

    imu = MPU6050()
    imu.start()

    try:
        for _ in range(20):
            time.sleep(0.5)
            print(json.dumps(imu.snapshot()), flush=True)

    except KeyboardInterrupt:
        pass

    finally:
        imu.close()