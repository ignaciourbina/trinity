"""Wire format shared by the server and the client.

Array payloads travel as ``.npz`` bytes (pickle disabled on load); small
control messages travel as JSON. All endpoints live under ``/v1``.
"""

from __future__ import annotations

import io

import numpy as np

API_PREFIX = "/v1"
NPZ_TYPE = "application/x-npz"
JSON_TYPE = "application/json"
MAX_BODY_BYTES = 256 << 20


def pack(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in arrays.items()})
    return buf.getvalue()


def unpack(data: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
