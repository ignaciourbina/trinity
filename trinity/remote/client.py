"""A Backend that forwards every call to a TrinityServer (e.g. one running in Colab)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Sequence

import numpy as np

from ..backends.base import Backend, ModelInfo, as_seqs
from .protocol import API_PREFIX, JSON_TYPE, NPZ_TYPE, pack, unpack


class RemoteError(RuntimeError):
    pass


class RemoteBackend(Backend):
    def __init__(self, url: str, token: str | None = None, timeout: float = 600.0,
                 retries: int = 3):
        self.url = url.rstrip("/")
        if self.url.endswith(API_PREFIX):
            self.url = self.url[: -len(API_PREFIX)]
        self.token = token or os.environ.get("TRINITY_TOKEN")
        self.timeout = timeout
        self.retries = retries
        self.info = ModelInfo(**self._call("/info", method="GET"))

    def _call(self, path: str, payload=None, method: str = "POST"):
        if isinstance(payload, bytes):
            data, ctype = payload, NPZ_TYPE
        elif payload is not None:
            data, ctype = json.dumps(payload).encode(), JSON_TYPE
        else:
            data, ctype = None, JSON_TYPE
        headers = {"Content-Type": ctype, "User-Agent": "trinity-client"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(self.url + API_PREFIX + path, data=data,
                                     headers=headers, method=method)
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read()
                    if resp.headers.get("Content-Type", "").startswith(NPZ_TYPE):
                        return unpack(body)
                    return json.loads(body)
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")
                # Client errors will not fix themselves; tunnels can hiccup with 502/503/504.
                if e.code < 500 or attempt == self.retries:
                    raise RemoteError(f"{path} failed with HTTP {e.code}: {detail}") from None
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                if attempt == self.retries:
                    raise RemoteError(f"cannot reach {self.url}: {e}") from None
            time.sleep(2 ** attempt)

    def encode(self, text: str) -> list[int]:
        return self._call("/encode", {"text": text})["ids"]

    def decode(self, ids: Sequence[int]) -> str:
        return self._call("/decode", {"ids": [int(i) for i in ids]})["text"]

    def topk_next(self, seqs, k, force_id=None):
        arrays = {"seqs": as_seqs(seqs), "k": k}
        if force_id is not None:
            arrays["force_id"] = force_id
        out = self._call("/topk", pack(**arrays))
        return out["ids"].astype(np.int64), out["logp"].astype(np.float64)

    def tv_at_positions(self, base, perturbed, positions):
        out = self._call("/tv", pack(base=np.asarray(base, dtype=np.int64),
                                     perturbed=as_seqs(perturbed),
                                     positions=np.asarray(positions, dtype=np.int64)))
        return out["tv"]

    def hidden_states(self, seqs, layers=None):
        arrays = {"seqs": as_seqs(seqs)}
        if layers is not None:
            arrays["layers"] = np.asarray(list(layers), dtype=np.int64)
        return self._call("/hidden", pack(**arrays))["h"]
