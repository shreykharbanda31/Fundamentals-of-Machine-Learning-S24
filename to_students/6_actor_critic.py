"""REINFORCE algorithm.

Author: Elie KADOCHE.
"""

import torch
from torch import optim as optim
from torch.distributions import Categorical
from src.envs.cartpole_v0 import CartpoleEnvV0
from src.envs.cartpole_v1 import CartpoleEnvV1
from src.models.actor_v0 import ActorModelV0
from src.models.actor_v1 import ActorModelV1
from src.models.critic import CriticModel
import torch.nn.functional as F

# Maximum environment length
HORIZON = 500

# ---> TODO: change the discount factor to solve the problem
DISCOUNT_FACTOR = 0.99

# Stopping criteria
MAX_EPISODES = 2000 #Early stopping
CONSECUTIVE_EPISODES = 10 # Criteria
TARGET_AVG_REWARD = 500 # Criteria 

# ---> TODO: change the learning rate to solve the problem
LEARNING_RATE = 0.001
CRITIC_LR = 0.001

if __name__ == "__main__":
    # Create environment and policy
    env = CartpoleEnvV0()
    actor = ActorModelV0()
    critic = CriticModel(input_size=env.observation_space.shape[0])
    actor_path = "/Users/shreykharbanda/Downloads/ML Spring 2024/to_students/saved_models/actor_3.pt"
    critic_path = "/Users/shreykharbanda/Downloads/ML Spring 2024/to_students/saved_models/critic_3.pt"

    # Training mode
    actor.train()
    critic.train()
    print(actor)
    print(critic)

    # Create optimizer with the policy parameters
    actor_optimizer = optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)

    # Run at max episodes
    training_iteration = 0
    consecutive_successes = 0 
    while True:

        # Experience
        # ------------------------------------------

        # Reset the environment
        state, _ = env.reset()

        # During experience, we will save:
        # - the probability of the chosen action at each time step pi(at|st)
        # - the rewards received at each time step ri
        saved_probabilities = list()
        saved_rewards = list()
        values = list()

        # Prevent infinite loop
        for t in range(HORIZON + 1):

            # Use the policy to generate the probabilities of each action
            probabilities = actor(state)

            # Create a categorical distribution over the list of probabilities
            # of actions and sample an action from it
            distribution = Categorical(probabilities)
            action = distribution.sample()

            # Forward pass through Critic
            value = critic(state)

            # Take the action
            state, reward, terminated, _, _ = env.step(action.item())

            # Save the probability of the chosen action and the reward
            saved_probabilities.append(probabilities[0][action])
            saved_rewards.append(reward)

            # Save the value from critic
            values.append(value)

            # End episode
            if terminated:
                break

        # Compute discounted sum of rewards
        # ------------------------------------------

        # Current discounted reward
        discounted_reward = 0.0

        # List of all the discounted rewards, for each time step
        discounted_rewards = list()


        for t in reversed(range(len(saved_rewards))):
            # Discounted reward at time t (G_t)
            discounted_reward = saved_rewards[t] + DISCOUNT_FACTOR * discounted_reward
            discounted_rewards.append(discounted_reward)
        # Reverse the rewards to achieve the right order
        discounted_rewards.reverse()
        # Eventually normalize for stability purposes
        discounted_rewards = torch.tensor(discounted_rewards)
        mean, std = discounted_rewards.mean(), discounted_rewards.std()
        discounted_rewards = (discounted_rewards - mean) / (std + 1e-7)
        values = torch.tensor(values)
        advantages = discounted_rewards - values

        # Update policy parameters
        # ------------------------------------------

        # For each time step
        actor_loss = list()
        for p, g in zip(saved_probabilities, advantages):

            # Compute policy loss
            time_step_actor_loss = -torch.log(p) * g  # negative log-likelihood * advantage estimate

            # Save it
            actor_loss.append(time_step_actor_loss)
        
        # Sum all the time step losses
        actor_loss = torch.cat(actor_loss).sum()

        # Reset gradients to 0.0
        actor_optimizer.zero_grad()

        # Compute the gradients of the loss (backpropagation)
        actor_loss.backward()

        # Update the policy parameters (gradient ascent)
        actor_optimizer.step()

        # Convert values to tensor
        values = torch.tensor(values, dtype=torch.float32, requires_grad=True)

        # Calculate Mean Squared Error loss between values and discounted rewards
        critic_loss = F.mse_loss(values, discounted_rewards)
        
        # Reset gradients to 0.0
        critic_optimizer.zero_grad()

        # Compute the gradients of the loss (backpropagation)
        critic_loss.backward()

        # Update the critic's parameters (gradient descent)
        critic_optimizer.step()
  

        # Logging
        # ------------------------------------------

        # Episode total reward
        episode_total_reward = sum(saved_rewards)

        # Aims to find consecutive training episode reward values greater than 500
        # the else statement ensures that only consecutive successes matter
        # This method is a simpler way to compare a window average with a target average without
        # explicitly calculating the window average each time we have a window
        if episode_total_reward >= TARGET_AVG_REWARD:
            consecutive_successes += 1
        else:
            consecutive_successes = 0

        # Check if consecutive successes reached
        if consecutive_successes >= CONSECUTIVE_EPISODES:
            print("Consecutive successful episodes reached. Training stopped.")
            break

        # Log results
        log_frequency = 5
        training_iteration += 1
        if training_iteration % log_frequency == 0:

            # Save neural network
            torch.save(actor, actor_path)
            torch.save(critic, critic_path)

            # Print results
            print("iteration {} - last reward: {:.2f}".format(
                training_iteration, episode_total_reward))

            # Early stopping as a preventive measure
            # Check if maximum number of episodes is reached
            if training_iteration >= MAX_EPISODES:
                print("Maximum number of episodes reached. Training stopped.")
                break
                
                
