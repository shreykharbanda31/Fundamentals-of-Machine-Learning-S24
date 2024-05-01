"""REINFORCE (with baseline) algorithm.

Author: Elie KADOCHE.
"""

import torch
from torch import optim as optim
from torch.distributions import Categorical

from src.envs.cartpole_v0 import CartpoleEnvV0
from src.models.actor_v0 import ActorModelV0
from src.models.critic import CriticModel

# Maximum environment length
HORIZON = 500

# ---> TODO: change the discount factor to solve the problem
DISCOUNT_FACTOR = 0.1

# ---> TODO: change the learning rate to solve the problem
LEARNING_RATE = 0.1

if __name__ == "__main__":
    # Create environment, policy and critic
    env = CartpoleEnvV0()
    actor = ActorModelV0()
    critic = CriticModel()
    actor_path = "./saved_models/actor_2.pt"
    critic_path = "./saved_models/critic_2.pt"

    # Training mode
    actor.train()
    critic.train()
    print(actor)
    print(critic)

    # Create optimizer with the policy parameters
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)

    # Create optimizer with the critic parameters
    critic_optimizer = optim.Adam(critic.parameters(), lr=LEARNING_RATE)

    # ---> TODO: based on the REINFORCE script, create the REINFORCE with
    # baseline script
