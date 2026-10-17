"""Command line: ``python -m trinity {toy,lm,serve} ...``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DEFAULT_PROMPTS, LMConfig, ToyConfig


def _add_backend_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("model")
    g.add_argument("--model", default="gpt2", help="HuggingFace model id (default: gpt2)")
    g.add_argument("--device", default="auto", help="auto, cuda, cpu, mps")
    g.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    g.add_argument("--batch-tokens", type=int, default=1 << 16,
                   help="token budget per forward pass (rows = budget // seq_len)")
    g.add_argument("--compile", action="store_true", help="torch.compile the transformer body")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trinity", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("toy", help="exact verification on the discrete toy model")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--t-max", type=int, default=10)
    p.add_argument("--out", default="outputs")
    p.add_argument("--no-plot", action="store_true")

    p = sub.add_parser("lm", aliases=["gpt2"], help="verification on a language model")
    _add_backend_args(p)
    p.add_argument("--backend", choices=["hf", "remote", "synthetic"],
                   help="default: remote if --remote is given, else hf")
    p.add_argument("--remote", metavar="URL", help="URL of a trinity server (e.g. one in Colab)")
    p.add_argument("--token", help="server token (or set TRINITY_TOKEN)")
    p.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--reward", default="fluency")
    p.add_argument("--telescope-depth", type=int, default=2)
    p.add_argument("--trajectories", type=int, default=15)
    p.add_argument("--perturbations", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--show", action="store_true", help="display figures (notebooks)")

    p = sub.add_parser("serve", help="serve a model to remote clients")
    _add_backend_args(p)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--token", help="shared secret (random if omitted)")
    p.add_argument("--tunnel", default="none", choices=["none", "cloudflared", "ngrok"])
    p.add_argument("--synthetic", action="store_true", help="serve the offline synthetic model")
    return parser


def _backend(args):
    from .backends import get_backend

    kind = args.backend or ("remote" if args.remote else "hf")
    return get_backend(kind, model=args.model, device=args.device, dtype=args.dtype,
                       batch_tokens=args.batch_tokens, compile=args.compile,
                       url=args.remote, token=args.token)


def cmd_toy(args) -> None:
    from .experiments import run_toy_suite
    from .report import print_toy

    res = run_toy_suite(ToyConfig(seed=args.seed, beta=args.beta, t_max=args.t_max))
    print_toy(res)
    if not args.no_plot:
        from .plotting import plot_toy
        print("\nFigures:", *map(str, plot_toy(res, args.out)))


def cmd_lm(args) -> None:
    from .experiments import run_lm_suite
    from .report import print_lm

    cfg = LMConfig(prompts=tuple(args.prompts), top_k=args.top_k, depth=args.depth, beta=args.beta,
                   reward=args.reward, telescope_depth=args.telescope_depth,
                   telescope_trajectories=args.trajectories,
                   influence_perturbations=args.perturbations, seed=args.seed)
    backend = _backend(args)
    print(f"Backend: {backend.info.name} on {backend.info.device}")
    res = run_lm_suite(backend, cfg)
    print_lm(res)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lm_results.json").write_text(json.dumps(res.to_dict(), indent=2))
    print(f"\nResults: {out / 'lm_results.json'}")
    if not args.no_plot:
        from .plotting import plot_lm
        print("Figures:", *map(str, plot_lm(res, out, show=args.show)))


def cmd_serve(args) -> None:
    from .backends import get_backend

    backend = get_backend("synthetic") if args.synthetic else get_backend(
        "hf", model=args.model, device=args.device, dtype=args.dtype,
        batch_tokens=args.batch_tokens, compile=args.compile)
    if args.tunnel != "none":
        from .remote.colab import serve
        if args.host != "127.0.0.1":
            raise SystemExit("--host cannot be combined with --tunnel (tunnels bind to localhost)")
        handle = serve(backend, port=args.port, tunnel=args.tunnel, token=args.token)
        try:
            handle.server._thread.join()
        except KeyboardInterrupt:
            handle.stop()
        return

    from .remote.server import TrinityServer
    server = TrinityServer(backend, args.host, args.port, args.token, verbose=True)
    print(f"Serving {backend.info.name} on {backend.info.device} at {server.address}\n"
          f"Token: {server.token}\n"
          f"Connect: python -m trinity lm --remote {server.address} --token {server.token}")
    server.serve_forever()


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    {"toy": cmd_toy, "lm": cmd_lm, "gpt2": cmd_lm, "serve": cmd_serve}[args.cmd](args)


if __name__ == "__main__":
    main()
