"""Seeded heavy-tailed synthetic returns and rolling risk forecast report."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .metrics import kupiec_pof_lr, rolling_var_es


def run(output_dir: Path, seed: int = 127) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    returns = rng.standard_t(df=5, size=800) / np.sqrt(5.0 / 3.0) * 0.01
    lookback, alpha = 250, 0.95
    var, es = rolling_var_es(returns, lookback=lookback, alpha=alpha)
    evaluation = np.arange(lookback, returns.size)
    exceptions = -returns[evaluation] > var[evaluation]
    summary = {
        "seed": seed,
        "lookback": lookback,
        "confidence_level": alpha,
        "evaluation_observations": int(evaluation.size),
        "exception_count": int(exceptions.sum()),
        "exception_rate": float(exceptions.mean()),
        "kupiec_unconditional_coverage_lr": kupiec_pof_lr(exceptions, alpha),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(evaluation, var[evaluation], label="Historical 95% loss VaR", color="#1d4ed8")
    ax.plot(evaluation, es[evaluation], label="Historical tail-mean ES", color="#7c3aed")
    ax.scatter(evaluation[exceptions], -returns[evaluation][exceptions],
               label="VaR exceptions", color="#dc2626", s=15)
    ax.set(title="Walk-forward tail-risk estimates on synthetic returns",
           xlabel="Synthetic return index", ylabel="Loss scale")
    ax.legend(frameon=False, fontsize=8)
    ax.text(0.01, 0.98, "SYNTHETIC • not a risk forecast for any real portfolio",
            transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "walk_forward_tail_risk.png", dpi=150)
    plt.close(fig)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=127)
    args = parser.parse_args()
    run(args.output_dir, seed=args.seed)
    print(f"Wrote synthetic tail-risk report to {args.output_dir}")


if __name__ == "__main__":
    main()

