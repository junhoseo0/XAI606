import warnings

with warnings.catch_warnings(record=True):
    # D4RL or its dependencies appear to reset the warning filter. Therefore,
    # instead of using `warnings.simplefilter("ignore")`, we use `record=True`
    # within the context manager. This approach prevents warnings from being
    # printed and effectiveky ignores them.
    import d4rl  # noqa: F401
import gym
import jax
import numpy as np
import hydra
from flax import nnx


class GaussianRandomPolicy(nnx.Module):
    def __init__(self, dim_state: int, dim_action: int, loc: float = 0.0, scale: float = 1.0, *, key):
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.loc = loc
        self.scale = scale
        self.key = key

    def __call__(self, state: jax.Array) -> jax.Array:
        self.key, subkey = jax.random.split(self.key)
        return jax.random.normal(subkey, (self.dim_action,), dtype=state.dtype) * self.scale + self.loc       


def rollout(cfg, env, policy, options: dict | None = None) -> list[float]:
    cumulative_rewards = []
    for _ in range(cfg.eval_episodes):
        cumulative_reward = 0.0
        state = env.reset(options=options)
        for _ in range(cfg.eval_steps):
            state, reward, done, _ = env.step(policy(action))
            cumulative_reward += reward

            if done:
                break
        cumulative_rewards.append(cumulative_reward)

    return cumulative_rewards


def evaluate(cfg, env, policy):
    perturbation_strengths = np.linspace(0.0, 100.0, 5)
    for perturbation in perturbation_strengths:
        cumulative_rewards = rollout(cfg, env, policy, options={"leg_joint_stiffness": perturbation})
        print(f"Perturbation: {perturbation:.2f} Return: {np.mean(cumulative_rewards):.2f}")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    env = gym.make(cfg.eval.env_id)
    dataset_env = gym.make(cfg.eval.dataset_id, max_episode_steps=cfg.eval.max_episode_steps)
    dataset = d4rl.qlearning_dataset(dataset_env)

    dim_state = env.observation_space.shape[-1]
    dim_action = env.action_space.shape[-1]

    policy_key = jax.random.PRNGKey(cfg.eval.seed)
    policy = GaussianRandomPolicy(dim_state, dim_action, key=policy_key)

    evaluate(env, policy)


if __name__ == "__main__":
    main()
