# Robot Package Delivery — Reinforcement Learning

A custom grid-world environment built with [Gymnasium](https://gymnasium.farama.org/) to experiment with tabular Q-learning. The agent controls a robot that must retrieve a package and deliver it to a designated destination while avoiding a hidden trap and completing the task within a step limit.

The project explores how reward design, exploration, grid size, and partial observability affect the agent's ability to learn a successful delivery policy.

## Project Overview

The environment consists of a two-dimensional grid containing:

- **Robot:** The agent's current position.
- **Package:** The item the robot must retrieve.
- **Delivery location:** The destination for the package.
- **Trap:** A hazardous location that terminates the episode if entered.

The robot can move in four directions: right, up, left, and down.

The agent observes the robot, package, and delivery locations, as well as whether it has picked up the package. The trap is hidden from the observation.

The objective is to pick up the package, deliver it successfully, avoid the trap, and complete the task before reaching the step threshold.

https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

A big part of the inspiration for the project is based on this link, and the instructions provided by gymnasium. 
As can be seen by my grid, which is entirely copied by the game of GridWorldEnv that can be viewed in the link.

## Technologies

- Python
- [Gymnasium](https://gymnasium.farama.org/) — custom reinforcement-learning environment
- NumPy — numerical operations and random sampling
- Tabular Q-learning — model-free reinforcement-learning algorithm

## Project Structure

```text
project/
├── agent/
│   └── q_learning.py
├── gymnasium_env/
│   ├── __init__.py
│   └── envs/
│       ├── __init__.py
│       └── grid_world.py
└── train.py
```

The structure is illustrative; filenames may differ depending on the current project layout.

## Environment Design

### Observation Space

The environment returns a dictionary containing:

- `robot_location`: Robot coordinates.
- `package_location`: Package coordinates.
- `delivery_location`: Delivery coordinates.
- `has_package`: Whether the robot is carrying the package.

The trap location is maintained internally and is not included in the observation.

The Q-learning agent converts each observation into a hashable tuple to use as a key in its Q-table.

### Action Space

The agent has four discrete actions, corresponding to the four movement directions.

### Episode Termination

An episode ends when the robot:

- Successfully delivers the package.
- Enters the trap.
- Reaches the configured step threshold.

The environment returns the observation, reward, termination status, truncation status, and an information dictionary in accordance with Gymnasium's API.

## Q-Learning

The agent uses tabular Q-learning to estimate the expected return of taking an action in a given state.

The Q-value update is:

\[
Q(s,a)\leftarrow Q(s,a)+\alpha
\left[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\right]
\]

where:

- \(Q(s,a)\) is the estimated value of an action.
- \(\alpha\) is the learning rate.
- \(r\) is the immediate reward.
- \(\gamma\) is the discount factor.
- \(s'\) is the next state.

The agent uses an epsilon-greedy policy: it explores randomly with probability epsilon and otherwise selects the action with the highest estimated Q-value. Epsilon decreases during training to shift gradually from exploration toward exploitation.

## Reward Design

The reward structure is intended to encourage successful delivery, discourage unnecessary movement, and penalize dangerous or unsuccessful episodes.

The current reward settings are configurable and have been adjusted experimentally. Values explored during development include:

| Event | Reward |
|---|---:|
| Ordinary step | -0.07 |
| Package pickup | +10 |
| Successful delivery | +33 |
| Trap collision | -10 |
| Step threshold reached | -7 |



These are experimental values, not a finalized or optimized reward function.

The reward settings influence the learned policy. In particular, a large trap penalty can encourage conservative behavior, while a large pickup reward relative to the delivery reward can make intermediate progress disproportionately attractive.

## Experimental Results

Training experiments have been conducted on 5×5 and 6×6 grids, with episodes limited by a configurable step threshold.

Observed results have varied between configurations. Examples include:

| Grid size | Approx. delivery success | Trap rate | Threshold rate |
|---|---:|---:|---:|
| 5×5 | 19.7% | 59.5% | 21.5% |
| 6×6 | 24.4% | 38.1% | 37.9% |

These figures are observations from particular training runs, not controlled benchmark results. The reward configuration and experimental conditions may differ between runs.

The results motivate further investigation into:

- How grid size affects trap avoidance and learning speed.
- How trap penalties affect delivery success.
- How the epsilon decay schedule affects exploration.
- How much training is needed for a stable policy.
- How performance changes across different trap locations and random seeds.

## Running the Project

Install the required dependencies:

```bash
pip install gymnasium numpy
```

Register the custom environment and create it:

```python
import gymnasium
import gymnasium_env

env = gymnasium.make("gymnasium_env/GridWorld-v0")
obs, info = env.reset()
```

The environment ID must match the registration in the package's `__init__.py`.

Run the training script:

```bash
python train.py
```

The training script initializes the Q-learning agent, runs multiple episodes, updates the Q-table, decays epsilon, and tracks delivery success, trap collisions, threshold terminations, and episode rewards.

Exact training parameters can be adjusted in the script.

## Rendering

Rendering can be enabled to inspect the robot's behavior visually. During training, rendering can be restricted to selected episodes to avoid slowing down the full training run.

A useful workflow is to train without rendering and then evaluate the learned policy with rendering enabled. During evaluation, exploration should be disabled and Q-values should not be updated.

For reproducible comparisons, keep the trap location and other experimental conditions consistent between runs.

## Current Limitations

- **Partial observability:** The agent cannot directly observe the trap location.
- **Tabular state space:** The number of possible states grows rapidly with grid size.
- **Limited generalization:** The Q-table learns values for specific observed states and may not generalize well to new configurations.
- **Reward sensitivity:** Changes to reward magnitudes can substantially alter the learned policy.
- **Experimental variability:** Random initial positions and trap placement can make comparisons between runs unreliable unless controlled.
- **Training efficiency:** A low delivery success rate may persist even after many episodes, particularly when the agent rarely discovers successful routes.

## Future Work

Potential next steps include:

- Compare reward configurations under controlled conditions.
- Evaluate policies over multiple random seeds.
- Record rolling success rates and average episode rewards.
- Compare against a random policy and a simple baseline navigation strategy.
- Test fixed trap locations to measure how reliably the agent learns to avoid hazards.
- Experiment with different epsilon-decay schedules.
- Explore alternative state representations or reinforcement-learning algorithms if the tabular approach becomes limiting.

## Learning Goals

This project serves as a practical exploration of:

- Reinforcement-learning fundamentals.
- Custom environment design with Gymnasium.
- Tabular Q-learning and epsilon-greedy exploration.
- Reward shaping and delayed rewards.
- Partial observability and state representation.
- Experimental evaluation and reproducibility.

The primary goal is not merely to maximize reward, but to understand how the environment and learning parameters influence the agent's behavior and its ability to complete the delivery task reliably.


