"""Expose a Backend over HTTP so a remote client can borrow its GPU.

Typical use is a Colab runtime serving GPT-2 through a tunnel (see ``colab.py``),
with the experiments running anywhere else through ``RemoteBackend``.
Every request must carry ``Authorization: Bearer <token>``.
"""

from __future__ import annotations

import hmac
import json
import secrets
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from ..backends.base import Backend
from .protocol import API_PREFIX, JSON_TYPE, MAX_BODY_BYTES, NPZ_TYPE, pack, unpack


class _HTTPError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def _check_ids(arr: np.ndarray, vocab: int, name: str) -> np.ndarray:
    if not np.issubdtype(arr.dtype, np.integer):
        raise _HTTPError(400, f"{name} must be an integer array")
    if arr.size and (arr.min() < 0 or arr.max() >= vocab):
        raise _HTTPError(400, f"{name} contains token ids outside [0, {vocab})")
    return arr.astype(np.int64, copy=False)


class TrinityServer:
    def __init__(self, backend: Backend, host: str = "127.0.0.1", port: int = 8765,
                 token: str | None = None, verbose: bool = False):
        self.backend = backend
        self.token = token or secrets.token_urlsafe(24)
        self.verbose = verbose
        self._lock = threading.Lock()  # one device, so serialise model calls
        self._httpd = ThreadingHTTPServer((host, port), self._handler_class())
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}"

    # --------------------------------------------------------------- routes
    def _route(self, path: str, body: bytes):
        b, vocab = self.backend, self.backend.info.vocab_size
        if path == "/info":
            return b.info.to_dict()
        if path == "/encode":
            return {"ids": b.encode(str(json.loads(body)["text"]))}
        if path == "/decode":
            ids = _check_ids(np.asarray(json.loads(body)["ids"], dtype=np.int64), vocab, "ids")
            return {"text": b.decode(ids.tolist())}

        req = unpack(body)
        if path == "/topk":
            force = int(req["force_id"]) if "force_id" in req else None
            k = int(req["k"])
            if not 1 <= k <= vocab:
                raise _HTTPError(400, "k out of range")
            with self._lock:
                ids, logp = b.topk_next(_check_ids(req["seqs"], vocab, "seqs"), k, force)
            return pack(ids=ids, logp=logp)
        if path == "/tv":
            base = _check_ids(req["base"], vocab, "base")
            perturbed = _check_ids(req["perturbed"], vocab, "perturbed")
            positions = req["positions"].astype(np.int64)
            if positions.size and (positions.min() < 0 or positions.max() >= base.shape[0]):
                raise _HTTPError(400, "positions out of range")
            with self._lock:
                return pack(tv=b.tv_at_positions(base, perturbed, positions))
        if path == "/hidden":
            layers = req["layers"].tolist() if "layers" in req else None
            with self._lock:
                return pack(h=b.hidden_states(_check_ids(req["seqs"], vocab, "seqs"), layers))
        raise _HTTPError(404, f"unknown endpoint {path}")

    def _handler_class(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                if server.verbose:
                    super().log_message(fmt, *args)

            def _send(self, status: int, payload):
                if isinstance(payload, bytes):
                    data, ctype = payload, NPZ_TYPE
                else:
                    data, ctype = json.dumps(payload).encode(), JSON_TYPE
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _handle(self, method: str):
                try:
                    if not self.path.startswith(API_PREFIX):
                        raise _HTTPError(404, "not found")
                    path = self.path[len(API_PREFIX):]
                    if path == "/health":
                        return self._send(200, {"ok": True})
                    auth = self.headers.get("Authorization", "")
                    if not hmac.compare_digest(auth.encode(), f"Bearer {server.token}".encode()):
                        raise _HTTPError(401, "missing or invalid token")
                    length = int(self.headers.get("Content-Length") or 0)
                    if length > MAX_BODY_BYTES:
                        raise _HTTPError(413, "request too large")
                    body = self.rfile.read(length) if length else b""
                    if method == "GET" and path != "/info":
                        raise _HTTPError(405, "use POST")
                    self._send(200, server._route(path, body))
                except _HTTPError as e:
                    self._send(e.status, {"error": str(e)})
                except (KeyError, ValueError) as e:
                    self._send(400, {"error": f"bad request: {e}"})
                except Exception as e:  # keep serving after unexpected failures
                    if server.verbose:
                        traceback.print_exc()
                    self._send(500, {"error": f"{type(e).__name__}: {e}"})

            def do_GET(self):
                self._handle("GET")

            def do_POST(self):
                self._handle("POST")

        return Handler

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "TrinityServer":
        """Serve from a daemon thread (non-blocking; suits notebooks)."""
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._httpd.server_close()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
