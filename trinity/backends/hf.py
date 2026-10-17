"""HuggingFace causal-LM backend (GPT-2 by default, any AutoModelForCausalLM works).

Performance notes:
* Sequences are processed in token-budgeted batches under ``torch.inference_mode``.
* The prompt shared by every node of a search tree is encoded once; its KV cache is
  broadcast across the batch so only the new suffix tokens are recomputed.
* The LM head is applied only at the positions we need (the last one, or the
  probe position for TV), never over the whole (batch, length, vocab) tensor.
* Top-k selection and EOS forcing happen on the device; only (N, k) arrays move to the host.
"""

from __future__ import annotations

import warnings
from typing import Iterator, Sequence

import numpy as np

from .base import Backend, ModelInfo, as_seqs

try:
    import torch
except ImportError as e:  # pragma: no cover - exercised only without torch
    raise ImportError("HFBackend needs torch and transformers: pip install 'trinity[hf]'") from e


def _resolve_device(device: str) -> "torch.device":
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class HFBackend(Backend):
    def __init__(self, model, tokenizer, device: str = "auto", batch_tokens: int = 1 << 16,
                 name: str | None = None, prefix_cache: bool = True):
        self.device = _resolve_device(device) if isinstance(device, str) else device
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.batch_tokens = batch_tokens
        self.prefix_cache = prefix_cache
        self._kv_key, self._kv = None, None
        self._base = model.base_model
        self._head = model.get_output_embeddings()
        # Models that post-process logits inside forward() (e.g. soft-capping) need the full path.
        self._full_forward = getattr(model.config, "final_logit_softcapping", None) is not None

        cfg = model.config
        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else cfg.eos_token_id
        self.info = ModelInfo(
            name=name or getattr(cfg, "_name_or_path", None) or cfg.model_type,
            vocab_size=int(self._head.out_features if hasattr(self._head, "out_features")
                           else cfg.vocab_size),
            eos_id=int(eos),
            hidden_dim=int(getattr(cfg, "n_embd", None) or cfg.hidden_size),
            n_layers=int(getattr(cfg, "n_layer", None) or cfg.num_hidden_layers),
            device=str(self.device),
        )

    @classmethod
    def from_pretrained(cls, name: str = "gpt2", device: str = "auto", dtype: str = "float32",
                        batch_tokens: int = 1 << 16, compile: bool = False) -> "HFBackend":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=_DTYPES[dtype])
        backend = cls(model, tokenizer, device=device, batch_tokens=batch_tokens, name=name)
        if compile:
            backend._base = torch.compile(backend._base, dynamic=True)
        return backend

    # ------------------------------------------------------------------ text
    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode([int(i) for i in ids])

    # --------------------------------------------------------------- helpers
    def _chunks(self, n: int, length: int) -> Iterator[slice]:
        rows = max(1, self.batch_tokens // max(length, 1))
        for start in range(0, n, rows):
            yield slice(start, min(start + rows, n))

    def _to_device(self, arr: np.ndarray) -> "torch.Tensor":
        return torch.from_numpy(np.ascontiguousarray(arr)).to(self.device, non_blocking=True)

    def _last_logits(self, x: "torch.Tensor", past=None) -> "torch.Tensor":
        """float32 logits at the last position of each row."""
        kw = dict(input_ids=x, past_key_values=past, use_cache=past is not None)
        if self._full_forward:
            return self.model(**kw).logits[:, -1].float()
        return self._head(self._base(**kw).last_hidden_state[:, -1]).float()

    # A search tree's sequences all share the prompt. Its KV cache is computed once and
    # broadcast to every row, so each forward pass only processes the new suffix tokens.
    def _prefix_cache(self, prefix: np.ndarray, rows: int):
        from transformers import DynamicCache

        key = prefix.tobytes()
        if self._kv_key != key:
            past = self._base(input_ids=self._to_device(prefix[None]), use_cache=True).past_key_values
            if hasattr(past, "layers"):
                self._kv = [(layer.keys, layer.values) for layer in past.layers]
            elif hasattr(past, "key_cache"):
                self._kv = list(zip(past.key_cache, past.value_cache))
            else:
                self._kv = [tuple(kv[:2]) for kv in past]
            self._kv_key = key
        cache = DynamicCache()
        for i, (k, v) in enumerate(self._kv):  # expand() is a view; nothing is copied here
            cache.update(k.expand(rows, -1, -1, -1), v.expand(rows, -1, -1, -1), i)
        return cache

    def _batch_logits(self, seqs: np.ndarray, prefix_len: int) -> "torch.Tensor":
        if prefix_len and self.prefix_cache:
            try:
                past = self._prefix_cache(seqs[0, :prefix_len], len(seqs))
                return self._last_logits(self._to_device(seqs[:, prefix_len:]), past)
            except Exception as e:  # unusual cache layouts: fall back to full recomputation
                warnings.warn(f"prefix KV cache disabled ({type(e).__name__}: {e})")
                self.prefix_cache = False
        return self._last_logits(self._to_device(seqs))

    # ------------------------------------------------------------ primitives
    @torch.inference_mode()
    def topk_next(self, seqs, k, force_id=None):
        seqs = as_seqs(seqs)
        ids_out = np.empty((len(seqs), k), dtype=np.int64)
        logp_out = np.empty((len(seqs), k), dtype=np.float64)
        shared = int(np.cumprod((seqs == seqs[0]).all(axis=0)).sum())
        prefix_len = min(shared, seqs.shape[1] - 1) if len(seqs) > 1 else 0
        for sl in self._chunks(len(seqs), seqs.shape[1]):
            logp = torch.log_softmax(self._batch_logits(seqs[sl], prefix_len), dim=-1)
            vals, ids = torch.topk(logp, k, dim=-1)
            if force_id is not None:
                missing = ~(ids == force_id).any(dim=-1)
                ids[missing, -1] = force_id
                vals[missing, -1] = logp[missing, force_id]
            ids_out[sl] = ids.cpu().numpy()
            logp_out[sl] = vals.double().cpu().numpy()
        return ids_out, logp_out

    @torch.inference_mode()
    def tv_at_positions(self, base, perturbed, positions):
        base = np.asarray(base, dtype=np.int64)
        perturbed = as_seqs(perturbed)
        positions = np.asarray(positions, dtype=np.int64)
        out = np.empty(len(positions))
        x_base = self._to_device(base[None])
        if self._full_forward:
            base_logits = self.model(input_ids=x_base, use_cache=False).logits[0]
        else:
            base_logits = self._head(self._base(input_ids=x_base, use_cache=False).last_hidden_state[0])
        base_p = torch.softmax(base_logits.float(), dim=-1)
        # Causal model: only the prefix up to t matters, so group rows by t and truncate.
        for t in np.unique(positions):
            rows = np.flatnonzero(positions == t)
            for sl in self._chunks(len(rows), t + 1):
                x = self._to_device(perturbed[rows[sl], : t + 1])
                p = torch.softmax(self._last_logits(x), dim=-1)
                out[rows[sl]] = (0.5 * (p - base_p[t]).abs().sum(-1)).cpu().numpy()
        return out

    @torch.inference_mode()
    def hidden_states(self, seqs, layers=None):
        seqs = as_seqs(seqs)
        layers = list(range(self.info.n_layers + 1)) if layers is None else list(layers)
        out = np.empty((len(seqs), len(layers), self.info.hidden_dim), dtype=np.float32)
        for sl in self._chunks(len(seqs), seqs.shape[1]):
            hs = self._base(input_ids=self._to_device(seqs[sl]), use_cache=False,
                            output_hidden_states=True).hidden_states
            out[sl] = torch.stack([hs[i][:, -1] for i in layers], 1).float().cpu().numpy()
        return out
