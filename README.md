# XAI606: Robust Offline RL as a Sequence Modeling Problem

## Overview of Robust Offline Reinforcement Learning (RL)

Deep RL is a subset of RL where both the policy and value function are approximated
using deep neural networks. The primary goal of deep RL is to learn a policy that
maximizes the cumulative sum of time-discounted rewards.

### What is Robust RL?

Robust RL focues on learning policies that can handle discrepencies between
training and testing environments, particularly when the transition dynamics in 
the test environment differ from those in the train environment. The aim is to
develope policies that are "robust", meaning that their performance deteriorates
more slowly compared to non-robust policies when facing such environmental
variations.

### What is Offline RL?

Offline RL is another branch of RL, where the agent learns from a fixed dataset,
collected beforehand using a behavior policy, without interacting with the
environment during training (as in the case in "online" RL).

### Combining Robustness and Offline Learning

Robust offline RL merges two challenges, requiring the agent to learn to adapt
to unseen environments, while trained with a fixed, finite dataset.

## Offline RL as a Sequence Modeling Problem

One approach to offline RL is to treat it as a sequence modeling task. In this
context, the policy predicts actions based on the histories of (return-to-go,
state, action) pairs. The return-to-go represents the expected future rewards
from the current state and encodes the trajectory's overall quality.

### Decision Transformer (DT)

DT leverages this sequence modeling approach by training a transformer model to
predict actions in supervised learning manner. DT has shown advantages in handling
long-term sparse reward problems and provides greater training stability by
avoiding the non-stationarity problem often encountered in RL.

However, applying DT-based model to robust RL tasks has its limitation. This 
project aims to explorer methods for improving DT-based model performance in
robust RL environments.

## Environments and Dataset

The target environment for this project is the MuJoCo environment provided
through the [D4RL](https://github.com/Farama-Foundation/D4RL) library. You
will use `halfcheetah-medium-v2` dataset which contains the transitions collected 
by a pre-trained Soft Actor-Critic (SAC) that was early-stopped in the training.

In this environment, the agent is rewarded base on its current speed, making
maximizing speed the primary objective.

Since RL generally does not use traditional validation/test data, the validation
and test sets here are presented as environments in which the agent can interact.
The validation environment conatins a limited range of perturbations, while the
test environment features a broader range of challenges.

### Perturbation Details

Perturbations occur in the stiffness of shin joint (`bshin_joint_stiffness`). In
the validation environment, the stiffness ranges from 180 to 90, while in the 
test environment, it ranges from 180 to 0.

## Evaluation Criteria

A trained policy will be evaluated based on three key criteria:

- Does it perform equivalently to a non-robust policy in an unperturbed environment?
- Does it outperform in environments with perturbations?
- Does it achieve the return-to-go target in the perturbed environment?

An example of how to evaluate a random policy in a perturbed environment is available
    in the `evaluate.py` file.
