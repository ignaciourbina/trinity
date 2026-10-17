"""Serve a model from a Google Colab runtime and make it reachable from anywhere.

In a Colab cell::

    from trinity.remote.colab import serve
    handle = serve("gpt2")          # prints the URL and token to use remotely

Then on any machine::

    python -m trinity lm --remote <URL> --token <TOKEN>

Tunnels:
* ``cloudflared`` (default): a free Cloudflare quick tunnel; no account needed.
* ``ngrok``: needs ``pip install pyngrok`` and an auth token (arg or NGROK_AUTHTOKEN).
* ``none``: no tunnel; only the runtime itself can reach the server.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from ..backends.base import Backend
from .server import TrinityServer

CLOUDFLARED_URL = ("https://github.com/cloudflare/cloudflared/releases/latest/download/"
                   "cloudflared-linux-{arch}")
# Quick-tunnel hosts look like https://word-word-word.trycloudflare.com; api.* is the control plane.
_TUNNEL_RE = re.compile(r"https://(?!api\.)[-a-z0-9]+\.trycloudflare\.com")


@dataclass
class ServeHandle:
    server: TrinityServer
    url: str
    token: str
    tunnel: subprocess.Popen | None = None

    def stop(self) -> None:
        if self.tunnel is not None:
            self.tunnel.terminate()
        self.server.stop()

    def client_command(self) -> str:
        return f"python -m trinity lm --remote {self.url} --token {self.token}"


def _cloudflared_binary() -> str:
    found = shutil.which("cloudflared")
    if found:
        return found
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine(), "amd64")
    path = Path.home() / ".cache" / "trinity" / "cloudflared"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CLOUDFLARED_URL.format(arch=arch), path)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def start_cloudflared(port: int, timeout: float = 60.0) -> tuple[str, subprocess.Popen]:
    proc = subprocess.Popen(
        [_cloudflared_binary(), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    found: list[str] = []
    failure: list[str] = []

    def pump():  # keep draining output so the pipe never blocks the tunnel
        for line in proc.stdout:
            if not found:
                m = _TUNNEL_RE.search(line)
                if m:
                    found.append(m.group(0))
                elif "failed to request quick Tunnel" in line:
                    failure.append(line.strip())

    threading.Thread(target=pump, daemon=True).start()
    deadline = time.time() + timeout
    while not found:
        if failure or proc.poll() is not None or time.time() > deadline:
            proc.terminate()
            raise RuntimeError("cloudflared did not open a tunnel"
                               + (f": {failure[0]}" if failure else ""))
        time.sleep(0.2)
    return found[0], proc


def start_ngrok(port: int, authtoken: str | None = None) -> str:
    from pyngrok import ngrok

    authtoken = authtoken or os.environ.get("NGROK_AUTHTOKEN")
    if authtoken:
        ngrok.set_auth_token(authtoken)
    return ngrok.connect(port, "http").public_url


def serve(model: str | Backend = "gpt2", *, port: int = 8765, tunnel: str = "cloudflared",
          token: str | None = None, ngrok_token: str | None = None, **backend_kwargs) -> ServeHandle:
    """Start a background server (and tunnel) for ``model``; returns immediately."""
    if isinstance(model, Backend):
        backend = model
    else:
        from ..backends.hf import HFBackend
        backend = HFBackend.from_pretrained(model, **backend_kwargs)

    server = TrinityServer(backend, host="127.0.0.1", port=port, token=token).start()
    proc = None
    try:
        if tunnel == "cloudflared":
            url, proc = start_cloudflared(port)
        elif tunnel == "ngrok":
            url = start_ngrok(port, ngrok_token)
        elif tunnel == "none":
            url = server.address
        else:
            raise ValueError(f"unknown tunnel {tunnel!r}; choose cloudflared, ngrok or none")
    except BaseException:
        server.stop()
        raise

    handle = ServeHandle(server, url, server.token, proc)
    print(f"Serving {backend.info.name} on {backend.info.device}\n"
          f"  URL:   {url}\n  Token: {handle.token}\n\n"
          f"Connect from any machine with:\n  {handle.client_command()}")
    return handle
