# SpaceRace RL Competition — Starting Kit

## Objective

Train a reinforcement learning agent to navigate the **SpaceRace** environment.
Your agent controls a ship that must reach the top row as many times as possible
before the 60-second timer runs out. The competition has four phases, each with
a harder difficulty level.



## Quick Start

```bash
pip install gymnasium numpy torch   # or your preferred RL library
python local_eval.py --difficulty 0  # runs your agent locally
```



## Files in this kit

| File | Description |
|---|---|
| `agent.py` | **Edit this.** Your agent submission. |
| `space_race_env.py` | The SpaceRace Gymnasium environment. Do not modify. |
| `local_eval.py` | Simulates the Codabench evaluator locally. Run before submitting. |
| `compute_baseline.py` | Computes the random-agent baseline score (professor use). |



## Environment

```python
from space_race_env import SpaceRaceEnv

env = SpaceRaceEnv(difficulty=0, round_time_seconds=60, ticks_per_second=10)
obs, info = env.reset()
```

**Observation** — `np.ndarray` of shape `(54, 39, 3)`, dtype `uint8`:
- RGB image of the game state (upsampled 3× from 18×13 grid)
- Background: dark blue `(5, 10, 20)`
- Debris: yellow/tan `(230, 180, 70)`
- Ship: cyan `(90, 220, 250)`

> **💡 Tip**: Normalize for neural networks: `obs_normalized = obs.astype(np.float32) / 255.0`

**Action space** — `Discrete(2)`:
- `0` → move up
- `1` → move down

**Reward** — cumulative over episode; main signal is `+1.0` per top-row crossing.

**Episode end** — `truncated=True` when the 60-second timer expires.


## Using Semantic Info for Heuristics (Training Only)

During **training**, you can access semantic information via `info["semantic_obs"]`:

```python
# Training environment (default)
env = SpaceRaceEnv(obs_mode="rgb")  # include_semantic_info=True by default
obs, info = env.reset()

semantic_obs = info["semantic_obs"]  # shape (18, 13, 3), float32
# Channel 0: ship position (1.0 at ship cell)
# Channel 1: debris positions (1.0 where debris)
# Channel 2: normalized time remaining
```

Use this to build heuristics that generate good training data for your neural network.

> **⚠️ Warning**: During Codabench evaluation, `include_semantic_info=False`. Accessing `info["semantic_obs"]` will raise a `KeyError`. Your agent's `select_action(obs)` must work with **only the RGB observation**.



## Difficulty levels

| Phase | Difficulty | What changes |
|---|---|---|
| 1 | 0 | Baseline — deterministic debris, normal speed |
| 2 | 1 | Faster debris |
| 3 | 2 | Random debris initialisation |
| 4 | 3 | Higher debris density + random init |



## How to Submit

1. **Implement your policy** in `agent.py` (see the `Agent` class).
2. **Test locally**: `python local_eval.py --difficulty 0`
3. **Package your submission**:

```bash
# Only agent.py is required.
# If you load a saved model, include the weights file too.
zip submission.zip agent.py                          # pure-Python agent
zip submission.zip agent.py model.pt                 # agent + PyTorch weights
```

4. Upload `submission.zip` on the Codabench competition page.



## Evaluation

Your agent is evaluated on **10 episodes** using fixed seeds (reproducible).
The leaderboard ranks by **mean score** (mean number of top-row crossings).

| Metric | Description |
|---|---|
| Mean Score | Average crossings over 10 episodes — **primary ranking metric** |
| Std Dev | Consistency of your agent |
| Min / Max | Best and worst single episode |



## Tips

- **Start simple**: even a hand-coded heuristic (move up unless blocked) beats random.
- **Frame stacking**: consider stacking the last 2-4 frames to capture debris motion.
- **Curriculum learning**: train on difficulty 0, then fine-tune on harder levels.
- **Test locally**: run `python local_eval.py` before submitting — it mirrors Codabench settings.




## Reference

- [Space Race (Wikipedia)](https://en.wikipedia.org/wiki/Space_Race_(video_game))
- [Gymnasium docs](https://gymnasium.farama.org/)
- [CleanRL](https://docs.cleanrl.dev/)
