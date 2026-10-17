# trinity

Numerical checks for *Autoregressive Language Models as Bayesian Optimal Controllers over
Hidden Markovian State Spaces* (`paper-2.tex`). The central identity is

    Γ = exp(A*/β) = π*/π_ref = dκ*/dκ_ref        [control] = [Bayes] = [HMM]

It is checked on an exact toy model and on real transformers (GPT-2 by default).
The model can run locally, or remotely on a Google Colab GPU.

## Quick start

```bash
pip install -e ".[hf]"              # drop [hf] on machines that only act as a remote client
python -m trinity toy               # exact toy model, runs in seconds
python -m trinity lm                # GPT-2 (downloads weights from the HF hub)
python -m trinity lm --backend synthetic   # offline: a small numpy LM, no download
pytest                              # tests use tiny random models, so nothing is downloaded
```

`trinity_sim.py` and `trinity_gpt2_colab.py` are thin launchers for the same commands.

## Using Colab

Open `notebooks/trinity_colab.ipynb` in Colab and select a GPU runtime. Then use either mode:

* **Run in Colab:** the experiments run on the Colab GPU and the figures show inline.
* **Colab as a remote GPU:** `serve("gpt2")` starts an authenticated server behind a Cloudflare
  quick tunnel (or ngrok) and prints a URL and a token. From any other machine run:

  ```bash
  python -m trinity lm --remote https://<words>.trycloudflare.com --token <TOKEN>
  ```

  The client sends token batches and gets back only top-K log-probs, TV distances or hidden
  states. One tree level costs one round trip, so a depth-3 run needs about four requests per prompt.

You can serve any machine with a GPU the same way: `python -m trinity serve --host 0.0.0.0 --port 8765`.

## Layout

| module | role |
| --- | --- |
| `trinity/identities.py` | the three formulas for Γ, plus Bellman, normalisation and decay-fit helpers |
| `trinity/rewards.py` | vectorised trajectory rewards; `REWARDS` registry |
| `trinity/tree.py` | soft DP over the top-K lookahead tree: one batched model call per depth level, log-space backward pass |
| `trinity/backends/` | `Backend` interface (`topk_next`, `tv_at_positions`, `hidden_states`) with HF, synthetic and remote implementations |
| `trinity/remote/` | HTTP server (bearer token, npz payloads), client, Colab tunnel helper |
| `trinity/experiments/` | toy and language-model suites, returning plain result objects |
| `trinity/report.py`, `trinity/plotting.py` | console tables and figures |

To add a model, implement `Backend`. To add a reward, write a function
`(tokens, logps, eos_id) -> rewards` and register it in `REWARDS`.

## Performance

* **Batched tree expansion.** The original code ran one forward pass per tree node with batch size 1.
  The tree is now expanded level by level, so depth 3 with K = 12 (1,464 nodes) takes four batched calls.
* **Prompt KV cache.** The prompt shared by every node is encoded once and its cache reused across
  the batch, so each node computes only its suffix tokens. Measured on 4 CPU cores with a
  GPT-2-small-shaped model, K = 12: depth 2 takes 0.87 s (the original took 6.98 s), and depth 3
  takes 8.6 s (58.5 s without the prefix cache).
* **Head only where needed.** The LM head runs only at the positions used. Top-K selection and EOS
  forcing happen on the device, so only (N, K) arrays come back to the host.
* **Influence decay** uses one call: rows are grouped by probe position and cut to the causal prefix.
* **Exact arithmetic.** The DP is computed in float64 log space, and every identity is checked at
  every node of the tree, not only at the root.
* **Tunable.** `--batch-tokens` sets memory use, `--dtype float16` and `--compile` trade
  precision for speed on GPUs.

## Notes

The rewrite fixed two problems in the original GPT-2 script: each child node's candidate token was
fed to the model twice, and sibling expansions wrote into a shared, mutable KV cache. Both corrupted
π_ref. The identities still held, since they hold for any π_ref, but the policies being tested were
not GPT-2's.
