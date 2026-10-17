"""
================================================================================
THE TRINITY: EMPIRICAL VERIFICATION ON GPT-2  (Colab entry point)
================================================================================
Verifies Gamma = exp(A*/beta) = pi*/pi_ref = dkappa*/dkappa_ref on a real
transformer. The implementation lives in the ``trinity`` package; this file
is a thin launcher that works both as a script and pasted into a Colab cell.

Run on the Colab GPU directly:
    !git clone https://github.com/ignaciourbina/trinity
    %cd trinity
    !pip install -q -e ".[hf]"
    !python trinity_gpt2_colab.py                    # same flags as `python -m trinity lm`

Use Colab as a remote GPU for a client running anywhere else:
    !python trinity_gpt2_colab.py --serve            # prints a public URL and token
    # elsewhere:  python -m trinity lm --remote <URL> --token <TOKEN>

See notebooks/trinity_colab.ipynb for the same steps as notebook cells.
================================================================================
"""

import sys

try:
    from trinity.cli import main
except ImportError:  # pasted into a fresh runtime: install the package first
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "trinity[hf] @ git+https://github.com/ignaciourbina/trinity"])
    from trinity.cli import main


if __name__ == "__main__":
    # Inside a notebook kernel sys.argv holds the kernel's own flags, not ours.
    argv = [] if "ipykernel" in sys.modules else sys.argv[1:]
    if argv[:1] == ["--serve"]:
        main(["serve", "--tunnel", "cloudflared", *argv[1:]])
    else:
        main(["lm", *argv, *(["--show"] if "ipykernel" in sys.modules else [])])
