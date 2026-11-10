"""Launch the Streamlit UI in a native Windows webview."""

from __future__ import annotations

import ctypes
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


APP_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
APP_SCRIPT = APP_DIR / "app.py"


def _show_error(message: str) -> None:
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(None, message, "Market Agent", 0x10)
    else:
        print(message, file=sys.stderr)


def _server_process(port: int) -> None:
    from streamlit.web import bootstrap

    options = {
        "global.developmentMode": False,
        "server.address": "127.0.0.1",
        "server.port": port,
        "server.headless": True,
        "server.fileWatcherType": "none",
        "browser.gatherUsageStats": False,
    }
    bootstrap.load_config_options(flag_options=options)
    bootstrap.run(
        str(APP_SCRIPT),
        False,
        [],
        options,
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until_ready(process: subprocess.Popen[bytes], port: int) -> None:
    health_url = f"http://127.0.0.1:{port}/_stcore/health"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("The app server stopped before it was ready.")
        try:
            with urlopen(health_url, timeout=1) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            time.sleep(0.25)
    raise RuntimeError("The app server did not start within 60 seconds.")


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--server-only":
        _server_process(int(sys.argv[2]))
        return

    try:
        import webview

        port = _free_port()
        if getattr(sys, "frozen", False):
            command = [sys.executable, "--server-only", str(port)]
        else:
            command = [sys.executable, str(Path(__file__).resolve()), "--server-only", str(port)]
        server = subprocess.Popen(
            command,
            cwd=APP_DIR,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            _wait_until_ready(server, port)
            webview.create_window(
                "Market Agent",
                f"http://127.0.0.1:{port}",
                width=1440,
                height=960,
                min_size=(960, 640),
            )
            webview.start()
        finally:
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    except Exception as exc:
        _show_error(f"Market Agent could not start.\n\n{exc}")


if __name__ == "__main__":
    main()