"""End-to-end smoke test of the spacerace_dqn package

Run with:
    python -m spacerace_dqn._smoke
"""
from __future__ import annotations

import shutil
from pathlib import Path

from .config import Config
from .evaluator import Evaluator
from .policies import (
    AlwaysUpPolicy,
    DQNInferencePolicy,
    RandomPolicy,
    RGBHeuristicPolicy,
)
from .preprocessing import FrameStacker
from .submission import SubmissionPackager, SUBMISSIONS_DIR
from .trainer import DQNTrainer


def main() -> None:
    cfg = Config(episodes=2, eval_every=1, baseline_episodes=2,
                 submission_name="task1_smoke", checkpoint_name="smoke.pt")

    evaluator = Evaluator(cfg)
    baselines = []
    for policy in (RandomPolicy(), AlwaysUpPolicy(), RGBHeuristicPolicy()):
        result = evaluator.run(policy, n_episodes=2)
        baselines.append(result)
        print(f" baseline {policy.name}: mean={result['mean_score']:.2f} "
              f"std={result['std_score']:.2f}")

    trainer = DQNTrainer(cfg)
    agent = trainer.train()
    print(f"  trainer: train_minutes={trainer.train_minutes:.2f}")

    final_eval = trainer.final_eval()
    print(f"  final_eval: mean_score={final_eval['mean_score']:.2f} "
          f"mean_max_q={final_eval['mean_max_q']:.3f}")

    packager = SubmissionPackager(cfg)
    summary = packager.package(
        agent,
        state_shape=trainer.state_shape,
        history=trainer.history,
        eval_history=trainer.eval_history,
        baselines=baselines,
        final_eval=final_eval,
        train_minutes=trainer.train_minutes,
    )
    print(f"  packaged: {summary['files']['zip']}")

    # Re-load the exported agent.py and re-evaluate
    agent_py = Path(summary["files"]["agent_py"])
    inference_policy = DQNInferencePolicy(agent_py)
    inference_eval = evaluator.run(inference_policy, n_episodes=2)
    print(f"  inference re-eval: mean={inference_eval['mean_score']:.2f}")

    shutil.rmtree(SUBMISSIONS_DIR / cfg.submission_name, ignore_errors=True)
    print("  smoke OK")


if __name__ == "__main__":
    main()
