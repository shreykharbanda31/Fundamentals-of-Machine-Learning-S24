import numpy as np
from collections import defaultdict
from copy import deepcopy


class Agent:
    """Agent interacting with the environment.
    
    Parameters
    ----------
    environment: 
        The environment.
    policy:
        Policy of the agent (default = random).
    player: 
        Player for games (1 or -1, default = 1).
    """
    
    def __init__(self, environment, policy=None, player=1):
        self.environment = environment
        self.player = player
        if policy is not None:
            self.policy = policy
        else:
            self.policy = self.random_policy
        
    def get_actions(self, state):
        """Get all possible actions."""
        if hasattr(self.environment, 'adversary'):
            # for games only
            return self.environment.get_actions(state, self.player)
        else:
            return self.environment.get_actions(state)
        
    def random_policy(self, state):
        """Random choice among possible actions."""
        actions = self.get_actions(state)
        if len(actions):
            probs = np.ones(len(actions)) / len(actions)
        else:
            probs = [1]
            actions = [None]
        return probs, actions

    def get_action(self, state):
        """Get selected action."""
        action = None
        probs, actions = self.policy(state)
        if len(actions):
            i = np.random.choice(len(actions), p=probs)
            action = actions[i]
        return action
    
    def get_episode(self, n_steps=100):
        """Get the states and rewards for an episode."""
        self.environment.reinit_state()
        state = deepcopy(self.environment.state)
        states = [state] # add the initial state
        rewards = [0]
        for t in range(n_steps):
            action = self.get_action(state)
            reward, stop = self.environment.step(action)
            state = deepcopy(self.environment.state)
            states.append(state)
            rewards.append(reward)
            if stop:
                break
        return stop, states, rewards
    
    def get_gains(self, n_steps=100, n_runs=100):
        """Get the gains (cumulative rewards) over independent runs."""
        gains = []
        for t in range(n_runs):
            _, _, rewards = self.get_episode(n_steps)
            gains.append(np.sum(rewards))
        return np.array(gains)
    
    
class OnlineEvaluation(Agent):
    """Online policy evaluation. The agent interacts with the environment and learns the value function of its policy.
    
    Parameters
    ----------
    environment: 
        The environment.
    policy:
        Policy of the agent (default = random).
    player: 
        Player for games (1 or -1, default = 1).
    gamma:
        Discount rate (in [0, 1], default = 1).
    n_steps:
        Number of steps per episode (default = 1000).
    """
    
    def __init__(self, environment, policy=None, player=1, gamma=1, n_steps=1000):
        super(OnlineEvaluation, self).__init__(environment, policy, player)   
        self.gamma = gamma 
        self.n_steps = n_steps 
        self.init_evaluation()

    def init_evaluation(self):
        self.state_value = defaultdict(int) # value of a state (0 if unknown)
        self.state_count = defaultdict(int) # count of a state (number of visits)
            
    def add_state(self, state):
        """Add a state if unknown."""
        state_code = self.environment.encode(state)
        if state_code not in self.state_value:
            self.state_value[state_code] = 0
            self.state_count[state_code] = 0
        
    def get_states(self):
        """Get known states."""
        states = [self.environment.decode(state_code) for state_code in self.state_value]
        return states
    
    def is_known(self, state):
        """Check if some state is known."""
        return self.environment.encode(state) in self.state_value

    def get_values(self, states):
        """Get the value function of some states.""" 
        state_codes = [self.environment.encode(state) for state in states]
        values = [self.state_value[state_code] for state_code in state_codes]
        return values
    
    def get_policy(self):
        """Get the best known policy."""
        best_action = defaultdict(lambda: None)
        states = self.get_states()
        for state in states:
            actions = self.get_actions(state)
            state_code = self.environment.encode(state)
            if len(actions) == 1:
                best_action[state_code] = 0
            elif len(actions) > 1:
                values_action = []
                for action in actions:
                    probs, states = self.environment.get_transition(state, action)
                    rewards = [self.environment.get_reward(state) for state in states]
                    values = self.get_values(states)
                    # expected value
                    value = np.sum(np.array(probs) * (np.array(rewards) + self.gamma * np.array(values)))
                    values_action.append(value)
                best_action[state_code] = np.argmax(self.player * np.array(values_action))
        def policy(state):
            actions = self.get_actions(state)
            state_code = self.environment.encode(state)
            if best_action[state_code] is not None:
                action = actions[best_action[state_code]]
                return [1], [action]
            else:
                if len(actions):
                    probs = np.ones(len(actions)) / len(actions)
                    return probs, actions
                else:
                    return [1], [None]
        return policy    
        
        
class OnlineControl(Agent):
    """Online control. The agent interacts with the environment and learns the best policy.
    
    Parameters
    ----------
    environment : 
        The environment.
    player : 
        Player for games (1 or -1, default = 1).
    gamma :
        Discount rate (in [0, 1], default = 1).
    n_steps :
        Number of steps per episode (default = 1000).
    eps: 
        Exploration rate (in [0, 1], default = 1). 
        Probability to select a random action.
    """
    
    def __init__(self, environment, player=1, gamma=1, n_steps=1000, eps=1):
        super(OnlineControl, self).__init__(environment, None, player)  
        self.gamma = gamma 
        self.n_steps = n_steps 
        self.eps = eps 
        self.init_evaluation()
                      
    def set_parameters(self, n_steps=None, eps=None):
        """Reset learning parameters."""
        if n_steps is not None:
            self.n_steps = n_steps
        if eps is not None:
            self.eps = eps
            
    def init_evaluation(self):
        """Init evaluation parameters."""
        self.state_action_value = defaultdict(lambda: defaultdict(int)) # value of a state-action pair (0 if unknown)  
        self.state_action_count = defaultdict(lambda: defaultdict(int)) # count of a state-action pair  
        
    def add_state_action(self, state, action):
        """Add a state-action pair if unknown."""
        state_code = self.environment.encode(state)
        if state_code not in self.state_action_value:
            self.state_action_value[state_code][action] = 0
            self.state_action_count[state_code][action] = 0
            
    def get_states(self):
        """Get known states."""
        states = [self.environment.decode(state_code) for state_code in self.state_action_value]
        return states
    
    def is_known(self, state):
        """Check if some state is known."""
        return self.environment.encode(state) in self.state_action_value
    
    def get_best_action(self, state):
        """Get the best action in some state.""" 
        actions = self.get_actions(state)
        state_code = self.environment.encode(state)
        values = self.player * np.array([self.state_action_value[state_code][action] for action in actions])
        i = np.random.choice(np.flatnonzero(values==np.max(values)))
        best_action = actions[i]
        return best_action

    def get_best_action_randomized(self, state):
        """Get the best action in some state, or a random state with probability epsilon.""" 
        if np.random.random() < self.eps:
            actions = self.get_actions(state)
            return actions[np.random.choice(len(actions))]
        else:
            return self.get_best_action(state)
        
    def get_policy(self):
        """Get the best known policy.""" 
        def policy(state):
            return [1], [self.get_best_action(state)]
        return policy
