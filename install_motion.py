"""Run once with the robot server stopped."""
import ast
import shutil
import time
from pathlib import Path


root = Path(__file__).resolve().parent
server = root / "backend" / "server.py"
dashboard = root / "dashboard" / "dashboard.html"

required_files = (
    server,
    dashboard,
    root / "backend" / "motion.py",
    root / "dashboard" / "motion.html",
)

for path in required_files:
    if not path.is_file():
        raise SystemExit("Missing file: " + str(path))

server_text = server.read_text(encoding="utf-8-sig")
dashboard_text = dashboard.read_text(encoding="utf-8-sig")

if "# MOTION_CALIBRATION_V1" in server_text:
    raise SystemExit(
        "Already installed; no files changed."
    )

tree = ast.parse(server_text)

names = {
    node.name
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
}

required_routes = {
    "api_drive",
    "api_stop",
    "api_test",
    "api_settings_set",
    "api_safety",
    "api_imu_calibrate",
    "api_imu_zero",
}

if not required_routes <= names:
    raise SystemExit(
        "Server version differs: required routes are missing. "
        "No files changed."
    )

marker = 'if __name__ == "__main__":'

if (
    server_text.count(marker) != 1
    or dashboard_text.count("<h2>Module tests</h2>") != 1
):
    raise SystemExit(
        "Unexpected file layout; no files changed."
    )

server_text = server_text.replace(
    marker,
    "# MOTION_CALIBRATION_V1\n"
    "from motion import install\n"
    "install(globals())\n\n"
    + marker,
)

dashboard_text = dashboard_text.replace(
    "<h2>Module tests</h2>",
    '<h2>Motion calibration</h2>\n'
    '<iframe src="/motion" title="Motion calibration" '
    'style="width:100%;height:740px;border:0"></iframe>\n'
    "<h2>Module tests</h2>",
)

ast.parse(server_text)
ast.parse(
    (root / "backend" / "motion.py").read_text(
        encoding="utf-8-sig"
    )
)

backup = root / (
    "backup_before_motion_"
    + time.strftime("%Y%m%d_%H%M%S")
)
backup.mkdir()

shutil.copy2(server, backup / "server.py")
shutil.copy2(dashboard, backup / "dashboard.html")

try:
    server.write_text(
        server_text,
        encoding="utf-8",
    )
    dashboard.write_text(
        dashboard_text,
        encoding="utf-8",
    )

except Exception:
    shutil.copy2(backup / "server.py", server)
    shutil.copy2(
        backup / "dashboard.html",
        dashboard,
    )
    raise

print("Installed. Backup:", backup)
print("Start: python3 backend/server.py")
print("Open Tests and refresh the page.")