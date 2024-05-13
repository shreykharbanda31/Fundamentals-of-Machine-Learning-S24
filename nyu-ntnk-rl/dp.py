#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 26, 2021
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from scipy import sparse


class PolicyEvaluation:
    """Evaluation of a policy by dynamic programming.
    
    Parameters
    ----------
    environment: 
        The environment.
    policy:
        Policy of the agent (default = random).
    player: 
        Player for games (1 or -1, default = 1).
    gamma:
        Discount factor (between 0 and 1).
    n_iter:
        Number of iterations of Bellman's equation.
    """
    
    def __init__(self, environment, policy, player=1, gamma=1, n_iter=100):
        self.environment = environment
        self.policy = policy
        self.player = player
        self.gamma = gamma
        self.n_iter = n_iter
        self.get_states()
        self.get_rewards()
        self.get_non_terminal()
        
    def get_states(self):
        """Index all states."""
        self.states = self.environment.get_states()
        self.n_states = len(self.states)
        self.state_id = {self.environment.encode(state): i for i, state in enumerate(self.states)}
        
    def get_state_id(self, state):
        return self.state_id[self.environment.encode(state)]

    def get_rewards(self):
        """Get the reward of each state."""
        rewards = np.zeros(self.n_states)
        for state in self.states:    
            i = self.state_id[self.environment.encode(state)]  
            rewards[i] = self.environment.get_reward(state)
        self.rewards = rewards
    
    def get_transitions(self):
        """Get the transition matrix (probability of moving from one state to another) in sparse format."""
        transitions = sparse.lil_matrix((self.n_states, self.n_states))
        for state in self.states:    
            i = self.state_id[self.environment.encode(state)]
            if not self.environment.is_terminal(state):
                for prob, action in zip(*self.policy(state)):
                    probs, states = self.environment.get_transition(state, action)
                    indices = np.array([self.state_id[self.environment.encode(s)] for s in states])
                    transitions[i, indices] += prob * np.array(probs)
        return sparse.csr_matrix(transitions)
    
    def get_non_terminal(self):
        """Get non-terminal states as a boolean vector."""
        transitions = self.get_transitions()
        self.non_terminal = transitions.indptr[1:] > transitions.indptr[:-1]
    
    def evaluate_policy(self):
        """Evaluate a policy by iteration of Bellman's equation."""
        transitions = self.get_transitions()
        values = np.zeros(self.n_states)
        for t in range(self.n_iter):
            values = transitions.dot(self.rewards + self.gamma * values)
        self.values = values
        return values
        
    def get_policy(self):
        """Get the best known policy."""
        best_actions = dict()
        for state in self.states: 
            i = self.state_id[self.environment.encode(state)]
            actions = self.environment.get_actions(state)
            values_actions = []
            for action in actions:
                probs, states = self.environment.get_transition(state, action)
                indices = np.array([self.state_id[self.environment.encode(s)] for s in states])
                value = np.sum(np.array(probs) * (self.rewards + self.gamma * self.values)[indices])
                values_actions.append(value)
            if len(values_actions):
                values = self.player * np.array(values_actions)
                top_actions = np.flatnonzero(values == values.max())
                best_actions[i] = actions[np.random.choice(top_actions)]
            else:
                best_actions[i] = None
        policy = lambda state: [[1], [best_actions[self.state_id[self.environment.encode(state)]]]]
        return policy

    @staticmethod
    def get_action(policy, state):
        """Action for a deterministic policy."""
        probs, actions = policy(state)
        return actions[0]        
    
    def is_same_policy(self, policy):
        """Test if the policy has changed."""
        for state in self.states:
            if self.get_action(policy, state) != self.get_action(self.policy, state):
                return False
        return True
