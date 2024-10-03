from collections import namedtuple

import d4rl  # noqa: F401
import d4rl.gym_mujoco  # noqa: F401
import gym
import jax
import numpy as np
import optax
from flax import nnx
from flax.training import checkpoints
from jax import numpy as jnp


class CausalAttentionBlock(nnx.Module):
    def __init__(self, emb_dim: int, num_heads: int, dropout_p: float = 0.1, *, rngs):
        self.num_heads = num_heads

        self.ln_1 = nnx.LayerNorm(emb_dim, rngs=rngs)
        self.kqv = nnx.Linear(emb_dim, 3 * emb_dim, rngs=rngs)
        self.lin = nnx.Linear(emb_dim, emb_dim, rngs=rngs)
        self.ln_2 = nnx.LayerNorm(emb_dim, rngs=rngs)

        self.mlp = nnx.Sequential(
            nnx.Linear(emb_dim, 4 * emb_dim, rngs=rngs),
            nnx.Linear(4 * emb_dim, emb_dim, rngs=rngs),
            nnx.gelu,
        )

    def __call__(self, x):
        B, T, C = x.shape

        causal_mask = nnx.make_causal_mask(jnp.zeros((B, T)))

        att = self.ln_1(x)
        q, k, v = jnp.split(self.kqv(att), 3, axis=2)
        k = k.reshape(B, T, self.num_heads, C // self.num_heads)
        q = q.reshape(B, T, self.num_heads, C // self.num_heads)
        v = v.reshape(B, T, self.num_heads, C // self.num_heads)
        att = nnx.dot_product_attention(q, k, v, mask=causal_mask).reshape(B, T, C)
        x = x + att
        x = x + self.mlp(self.ln_2(x))
        return x


class DecisionGPT(nnx.Module):
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        emb_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        max_timesteps: int = 1000,
        dropout_p: float = 0.1,
        *,
        rngs,
    ):
        self.obs_size = obs_size
        self.act_size = act_size
        self.emb_dim = emb_dim

        self.embed_obs = nnx.Linear(obs_size, emb_dim, rngs=rngs)
        self.embed_act = nnx.Linear(act_size, emb_dim, rngs=rngs)
        self.embed_return = nnx.Linear(1, emb_dim, rngs=rngs)
        self.embed_timestep = nnx.Embed(max_timesteps, emb_dim, rngs=rngs)

        self.transformer = nnx.Sequential(
            *[
                CausalAttentionBlock(emb_dim, num_heads, rngs=rngs)
                for _ in range(num_layers)
            ]
        )
        self.ln = nnx.LayerNorm(emb_dim, rngs=rngs)

        self.action_head = nnx.Sequential(
            nnx.Linear(emb_dim, act_size, rngs=rngs), nnx.tanh
        )

    def __call__(self, obss, actions, return_to_gos, timesteps):
        B, T = obss.shape[:2]

        obs_embeddings = self.embed_obs(obss)
        act_embeddings = self.embed_act(actions)
        return_embeddings = self.embed_return(return_to_gos)
        timestep_embeddings = self.embed_timestep(timesteps)

        obs_embeddings += timestep_embeddings
        act_embeddings += timestep_embeddings
        return_embeddings += timestep_embeddings

        seqs = jnp.stack([return_embeddings, obs_embeddings, act_embeddings], axis=1)
        seqs = jnp.permute_dims(seqs, (0, 2, 1, 3)).reshape(B, 3 * T, self.emb_dim)

        x = self.transformer(seqs)
        x = self.ln(x)

        x = x.reshape(B, T, 3, self.emb_dim)
        x = jnp.permute_dims(x, (0, 2, 1, 3))
        return self.action_head(x[:, 1])


def loss_fn(model: DecisionGPT, batch):
    actions = model(
        batch["observations"],
        batch["actions"],
        batch["return_to_gos"],
        batch["timesteps"],
    )
    loss = ((actions - batch["actions"]) ** 2).mean()
    return loss


@nnx.jit
def train_step(
    model: DecisionGPT, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch
):
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    metrics.update(values=loss)
    optimizer.update(grads)


@nnx.jit
def get_action(model: DecisionGPT, states, actions, return_to_gos, timesteps):
    return model(states, actions, return_to_gos, timesteps)


def rollout(
    env,
    model: DecisionGPT,
    target_return: int,
    eval_episodes: int = 10,
    eval_steps: int = 1000,
    options: dict | None = None,
) -> list[float]:
    cumulative_rewards = []
    for _ in range(eval_episodes):
        cumulative_reward = 0.0
        state = env.reset(options=options)

        states = jnp.asarray([state])
        actions = jnp.zeros((0, model.act_size))
        return_to_gos = jnp.array([[target_return]])
        timesteps = jnp.array([[0]])

        for t in range(eval_steps):
            actions = jnp.concat([actions, jnp.zeros((1, model.act_size))], axis=0)
            action = get_action(model, states, actions, return_to_gos, timesteps)
            actions.at[-1].set(action)

            state, reward, done, _ = env.step(action)
            states = jnp.concat([states, jnp.asarray([state])], axis=0)
            return_to_gos = jnp.concat(
                [return_to_gos, jnp.array([[return_to_gos[-1] - reward]])]
            )
            timesteps = jnp.concat([timesteps, jnp.full((1, 1), t)])

            cumulative_reward += reward

            if done:
                break
        cumulative_rewards.append(cumulative_reward)

    return cumulative_rewards


def main():
    env = gym.make("halfcheetah-medium-v2")
    dataset = env.get_dataset()
    dataset = {
        k: v
        for k, v in dataset.items()
        if not (k.startswith("infos") or k.startswith("metadata"))
    }

    sequences = list(d4rl.sequence_dataset(env, dataset=dataset))
    dataset_seq = {
        key: jnp.stack([seq[key] for seq in sequences]) for key in sequences[0].keys()
    }

    model = DecisionGPT(
        obs_size=env.observation_space.shape[0],
        act_size=env.action_space.shape[0],
        rngs=nnx.Rngs(0),
    )

    checkpoints.save_checkpoint("/workspaces/XAI606/models", nnx.state(model), step=0)
    metrics = nnx.metrics.Average()
    optimizer = nnx.Optimizer(model, optax.adam(3e-4))

    rollout_cfg_type = namedtuple("RolloutConfig", ["eval"])
    eval_cfg_type = namedtuple(
        "EvalConfig", ["env_id", "dataset_id", "max_episode_steps", "seed"]
    )
    cfg = rollout_cfg_type(
        eval=eval_cfg_type(
            env_id="halfcheetah-medium-v2",
            dataset_id="halfcheetah-medium-v2",
            max_episode_steps=1000,
            seed=0,
        )
    )

    batch_size = 16
    key = jax.random.PRNGKey(0)
    for step in range(1_000_000):
        key, sample_key = jax.random.split(key)
        rand_idx = jax.random.randint(sample_key, (batch_size,), 0, len(sequences))
        batch = {k: v[rand_idx] for k, v in dataset_seq.items()}

        batch["return_to_gos"] = batch["rewards"].cumsum(axis=-1)[:, ::-1, None]
        batch["rewards"] = batch["rewards"][:, :, None]
        batch["timesteps"] = jnp.tile(
            jnp.arange(start=0, stop=batch["rewards"].shape[1]), (batch_size, 1)
        )
        train_step(model, optimizer, metrics, batch)

        if step > 0 and step % 1_000 == 0:
            loss = metrics.compute()
            metrics.reset()
            print(f"Step {step}, Loss: {loss}")

        if step > 0 and step % 10_000 == 0:
            model.eval()
            cum_rews = rollout(cfg, env, model)
            print(f"Step {step}, Cumulative rewards: {cum_rews}")
            model.train()

            checkpoints.save_checkpoint(
                "/workspaces/XAI606/models", nnx.state(model), step=step
            )

    perturbations = np.linspace(0.0, 100.0, 5)
    for perturbations in perturbations:
        cumulative_rewards = rollout(
            env, model, perturbations, options={"bshin_joint_stiffness": perturbations}
        )
        print(
            f"Perturbation: {perturbations:.2f} Return: {np.mean(cumulative_rewards):.2f}"
        )


if __name__ == "__main__":
    main()
