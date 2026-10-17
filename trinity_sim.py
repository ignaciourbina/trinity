"""
=============================================================================
COMPUTATIONAL SANITY CHECK: THE TRINITY  (toy model entry point)
=============================================================================
Exact verification on a discrete model (8 states, tokens {a, b, c, EOS},
deterministic transitions, softmax reference policy, endogenous stopping).
The implementation lives in ``trinity.experiments.toy``; equivalent to
``python -m trinity toy``.
"""

import sys

from trinity.cli import main

if __name__ == "__main__":
    main(["toy", *sys.argv[1:]])
