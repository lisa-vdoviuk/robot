"""MPU6050 telemetry only: sensor axes, no navigation or motor control."""
import math
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
        accel_g=[round(v, 4) for v in accel],
        gyro_dps=[round(v, 3) for v in gyro],
        temp_c=round(temperature / 340.0 + 36.53, 2),
        accel_norm_g=round(norm, 4),
    )


class MPU6050:
    def __init__(self, bus_number=1):
        self.bus_number = bus_number
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None

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
        )

    def publish(self, **values):
        with self.lock:
            self.state.update(values)

    def snapshot(self):
        with self.lock:
            result = dict(self.state)

        ts = result["ts_mono"]
        age = None if ts is None else max(0, time.monotonic() - ts)

        result["age_ms"] = (
            None if age is None else round(age * 1000)
        )
        result["valid"] = bool(
            result["valid"] and age is not None and age <= 0.5
        )

        if not result["valid"]:
            for key in (
                "accel_g",
                "gyro_dps",
                "temp_c",
                "accel_norm_g",
            ):
                result[key] = None

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

            # PLL X gyro clock, all axes enabled, DLPF, 50 Hz.
            # Measurement ranges: +/-250 deg/s and +/-2 g.
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
                        "MPU6050 configuration mismatch "
                        f"at 0x{register:02x}"
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
                error=(
                    "Install python3-smbus "
                    "(or smbus2 in your venv)"
                ),
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

    def test(self):
        first = self.snapshot()
        time.sleep(0.3)
        current = self.snapshot()

        if not current["valid"]:
            return False, (
                current["error"]
                or "No fresh MPU6050 samples; check I2C, wiring and power"
            )

        if current["seq"] <= first["seq"]:
            return False, "MPU6050 samples are not updating"

        norm = current["accel_norm_g"]

        if not 0.5 <= norm <= 1.5:
            return False, (
                f"Fresh data, but |a|={norm:.3f} g; "
                "keep robot still and repeat"
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