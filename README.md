# The Trinity: Empirical Verification on GPT-2

This project empirically verifies **The Trinity identity** on a real transformer (GPT-2 124M):

\[
\Gamma = \exp(A^*/\beta) = \pi^*/\pi_{\text{ref}} = \frac{d\kappa^*}{d\kappa_{\text{ref}}}
\]

The main script is `trinity_gpt2_colab.py` (originally written for Colab, now runnable locally on CPU as well).

## Quick Start

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python trinity_gpt2_colab.py
```

Or run end-to-end with:

```bash
bash run.sh
```

## What the Script Verifies

- Loads GPT-2 and computes recursive partition values \(Z(h_t)\)
- Verifies Γ equivalence across:
  - Optimal control (`exp(A*/beta)`)
  - Bayesian policy ratio (`pi*/pi_ref`)
  - HMM kernel ratio (`dκ*/dκ_ref`)
- Checks telescoping identity over sampled trajectories
- Measures influence decay under token perturbations
- Produces a multi-panel verification figure

## Output Files

Running the script generates:

- `trinity_gpt2_verification.png`
- `trinity_gpt2_verification.pdf`

## Runtime Notes

- CPU: typically ~10 minutes (with CPU-friendly defaults in this repo, often ~5–8 min)
- GPU (e.g., Colab T4): typically ~2–3 minutes
