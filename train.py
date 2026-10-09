import gymnasium
import gymnasium_env

from agent.q_learning import QLearning

# Add render_mode = "human" to add visualization
env = gymnasium.make(
    "gymnasium_env/GridWorld-v0",
    render_mode = "human",
    )

q_learning = QLearning(
    env=env,
    learning_rate = 0.1,
    initial_epsilon = 1,
    epsilon_decay = 0.0005,
    final_epsilon = 0.05,
    discount_factor = 0.95,
)

episodes = 10000

successful_deliveries = 0
trap_deaths = 0
step_threshold_reached = 0


for episode in range(episodes):
    env.unwrapped.render_enabled = episode >= episodes - 10
    
    obs, info = env.reset()

    terminated = False
    truncated = False
    episode_reward = 0

    while not terminated and not truncated:
        
        state = q_learning.get_state(obs)
        action = q_learning.get_action(state)

        next_obs, reward, terminated, truncated, info = env.step(action)   
        next_state = q_learning.get_state(next_obs)

        q_learning.update(state, action, reward, next_state, terminated)

        episode_reward += reward
        obs = next_obs

    if info["delivery_succeeded"]:
        successful_deliveries += 1

    if info["hit_trap"]:
        trap_deaths += 1

    if info["step_threshold_reached"]:
        step_threshold_reached += 1

    q_learning.decay_epsilon()

    completed_episodes = episode + 1

    success_rate = successful_deliveries / completed_episodes
    trap_rate = trap_deaths / completed_episodes
    step_threshold_rate = step_threshold_reached / completed_episodes

    print(
    f"Episode: {episode}. "
    f"Reward: {episode_reward:.2f}. "
    f"Epsilon: {q_learning.epsilon:.3f}. "
    f"Success rate: {success_rate:.1%}. "
    f"Trap rate: {trap_rate:.1%}. "
    f"Threshold reached: {step_threshold_rate:.1%}."
    )


env.close()

