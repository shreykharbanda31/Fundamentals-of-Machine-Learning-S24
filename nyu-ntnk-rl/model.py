#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Nov 21, 2020
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
import itertools

from agent import Agent
from display import display_position, display_board


class Environment:
    """Generic environment. The reward only depends on the new state."""
    def __init__(self):
        self.state = self.init_state()

    @staticmethod
    def init_state():
        return None

    def reinit_state(self):
        self.state = self.init_state()  
    
    @staticmethod
    def get_states():
        raise ValueError("Not available. The state space might be too large.")
    
    @staticmethod
    def get_actions(state):
        return []

    @staticmethod
    def get_transition(state, action):
        probs = [1]
        states = [deepcopy(state)]
        return probs, states

    @staticmethod
    def get_reward(state):
        return None

    @staticmethod
    def is_terminal(state):
        return False

    @staticmethod
    def encode(state):
        """Encoder, making state hashable."""
        return state
    
    @staticmethod
    def decode(state):
        """Decoder."""
        return state
    
    def get_model(self, state, action):
        """Get the model from a given state and action (transition probabilities, new states and rewards)."""
        probs, states = self.get_transition(state, action)
        rewards = [self.get_reward(state) for state in states]
        return probs, states, rewards
    
    def step(self, action):
        """Apply action, get reward and modify state. Check whether the new state is terminal."""
        reward = 0
        stop = True
        if action is not None and action in self.get_actions(self.state):
            probs, states, rewards = self.get_model(self.state, action)
            i = np.random.choice(len(probs), p=probs)
            state = states[i]
            self.state = state
            reward = rewards[i]
            stop = self.is_terminal(state)
        return reward, stop

    def display(self, states=None):
        """Display current states or animation of sequence of states if provided."""
        return None
                
                
class Walk(Environment):
    """Walk."""

    Size = (7, 7)
    Rewards = {(1, 1): 1, (1, 5): 2, (5, 5): 3, (5, 1): 4}

    def __init__(self):
        super(Walk, self).__init__()    

    @classmethod
    def set_parameters(cls, size, rewards):
        cls.Size = size
        cls.Rewards = rewards

    @staticmethod
    def init_state():
        return np.array([0, 0])
    
    @staticmethod
    def is_valid(state):
        n, m = Walk.Size
        x, y = tuple(state)
        return 0 <= x < n and 0 <= y < m
 
    @staticmethod
    def get_states():
        n, m = Walk.Size
        states = [np.array([x,y]) for x in range(n) for y in range(m)]
        return states
    
    @staticmethod
    def encode(state):
        return tuple(state)
    
    @staticmethod
    def decode(state):
        return np.array(state)
    
    @staticmethod
    def get_actions(state):
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        actions = [action for action in moves if Walk.is_valid(state + action)]
        return actions

    @staticmethod
    def get_transition(state, action):
        probs = [1]
        states = [state.copy() + action]
        return probs, states
 
    @staticmethod
    def get_reward(state):
        reward = 0
        if tuple(state) in Walk.Rewards:
            reward = Walk.Rewards[tuple(state)]
        return reward

    def display(self, states=None, marker='o', marker_size=200, marker_color='b', interval=200):
        shape = (*self.Size, 3)
        image = 200 * np.ones(shape).astype(int)
        return display_position(image, self.state, states, marker, marker_size, marker_color, interval)

    @staticmethod
    def display_values(values):
        image = np.zeros(Walk.Size)
        values_scaled = np.array(values)
        values_scaled -= np.min(values)
        if np.max(values_scaled) > 0:
            values_scaled /= np.max(values_scaled)
        states = Walk.get_states()
        for state, value in zip(states, values_scaled):
            image[tuple(state)] = 1 - 0.8 * value 
            plt.imshow(image, cmap='gray');
            plt.axis('off')

    @staticmethod
    def display_policy(policy):
        image = np.zeros(Walk.Size)
        plt.imshow(image, cmap='gray')
        states = Walk.get_states()
        for state in states:
            _, actions = policy(state)
            action = actions[0]
            if action == (0, 0):
                plt.scatter(state[1], state[0], s=100, c='b')
            else:
                plt.arrow(state[1], state[0] , action[1], action[0], color='r', width=0.15, length_includes_head=True)
        plt.axis('off')


class Maze(Environment):
    """Maze."""

    Map = np.ones((2, 2)).astype(int)
    Start_State = (0, 0)
    Exit_States = [(1, 1)]

    def __init__(self):
        super(Maze, self).__init__()

    @classmethod
    def set_parameters(cls, maze_map, start_state, exit_states):
        cls.Map = maze_map
        cls.Start_State = start_state
        cls.Exit_States = exit_states

    @staticmethod
    def init_state():
        return np.array(Maze.Start_State)

    @staticmethod
    def is_valid(state):
        n, m = Maze.Map.shape
        x, y = tuple(state)
        return 0 <= x < n and 0 <= y < m and Maze.Map[x, y]
    
    @staticmethod
    def get_states():
        n, m = Maze.Map.shape
        states = [np.array([x, y]) for x in range(n) for y in range(m) if Maze.is_valid(np.array([x, y]))]
        return states
    
    @staticmethod
    def encode(state):
        return tuple(state)
    
    @staticmethod
    def decode(state):
        return np.array(state)

    @staticmethod
    def get_actions(state):
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        actions = []
        for move in moves:
            if Maze.is_valid(state + move):
                actions.append(move)
        return actions

    @staticmethod
    def get_transition(state, action):
        probs = [1]
        states = [state.copy() + action]
        return probs, states

    @staticmethod
    def get_reward(state):
        return -1

    @staticmethod
    def is_terminal(state):
        return tuple(state) in Maze.Exit_States

    def display(self, states=None, marker='o', marker_size=200, marker_color='b', interval=200):
        shape = (*Maze.Map.shape, 3)
        image = np.zeros(shape).astype(int)
        for i in range(3):
            image[:, :, i] = 255 * Maze.Map
        return display_position(image, self.state, states, marker, marker_size, marker_color, interval)

    @staticmethod
    def display_values(values):
        image = np.zeros(Maze.Map.shape)
        values_scaled = np.array(values)
        if np.min(values_scaled):
            values_scaled /= np.min(values_scaled)
        if np.max(values_scaled) > 0:
            values_scaled /= np.max(values_scaled)
        states = Maze.get_states()
        for state, value in zip(states, values_scaled):
            image[tuple(state)] = 1 - 0.8 * value 
            plt.imshow(image, cmap='gray');
            plt.axis('off')

    @staticmethod
    def display_policy(policy):
        image = np.zeros(Maze.Map.shape)
        states = Maze.get_states()
        for state in states:
            image[tuple(state)] = 1
        plt.imshow(image, cmap='gray')
        for state in states:
            if not Maze.is_terminal(state):
                _, actions = policy(state)
                action = actions[0]
                if action == (0,0):
                    plt.scatter(state[1], state[0], s=100, c='b')
                else:
                    plt.arrow(state[1], state[0] , action[1], action[0], color='r', width=0.15, length_includes_head=True)
        plt.axis('off')
        
    
class Game(Environment):
    """Generic 2-player game. The adversary is part of the environment."""
    
    def __init__(self, adversary_policy=None, player=1, play_first=True):
        if play_first:
            self.first_player = player
        else:
            self.first_player = -player
        self.player = player
        self.adversary = Agent(self, adversary_policy, player=-player)
        super(Game, self).__init__()    
        
    def is_terminal(self, state):
        return bool(self.get_reward(state)) or not len(self.get_actions(state))

    def step(self, action=None):
        reward = 0
        stop = True
        if not self.is_terminal(self.state):
            player, board = self.state
            if player == -self.player:
                action = self.adversary.get_action(self.state)
            if action is not None:
                probs, states, rewards = self.get_model(self.state, action)
                i = np.random.choice(len(probs), p=probs)
                state = states[i]
                self.state = state
                reward = rewards[i]
                stop = self.is_terminal(state)
        return reward, stop
    
        
class TicTacToe(Game):
    """Tic-tac-toe game."""

    def init_state(self):
        board = np.zeros((3, 3)).astype(int)
        return [self.first_player, board]

    @staticmethod
    def encode(state):
        player, board = state
        if player == 1:
            return b'\x01' + board.tobytes()
        else:
            return b'\x00' + board.tobytes()
        
    @staticmethod
    def decode(state_code):
        player_code = state_code[0]
        if player_code:
            player = 1
        else:
            player = -1
        board_code = state_code[1:]
        board = np.frombuffer(board_code, dtype=int).reshape((3,3))
        return player, board
    
    def is_valid(self, state):
        player, board = state
        sums = set(board.sum(axis=0)) | set(board.sum(axis=1))
        sums.add(board.diagonal().sum())
        sums.add(np.fliplr(board).diagonal().sum())
        if 3 in sums and -3 in sums:
            return False
        if player == self.first_player:
            return np.sum(board==player) == np.sum(board==-player)
        else:
            return np.sum(board==player) == np.sum(board==-player)-1

    def is_terminal(self, state):
        player, board = state
        sums = set(board.sum(axis=0)) | set(board.sum(axis=1))
        sums.add(board.diagonal().sum())
        sums.add(np.fliplr(board).diagonal().sum())
        return 3 in sums or -3 in sums or board.astype(bool).all()
        
    def get_states(self):
        boards = [np.array(board).reshape(3,3) - 1 for board in itertools.product(range(3), repeat=9)]
        states = [(1, board) for board in boards] + [(-1, board) for board in boards]
        states = [state for state in states if self.is_valid(state)]
        return states
    
    def get_actions(self, state, player=None):
        actions = []
        if not self.is_terminal(state):
            if player is None:
                player = self.player
            current_player, board = state
            if player == current_player:
                x_, y_ = np.where(board == 0)
                actions = [(x, y) for x, y in zip(x_, y_)]
            else:
                actions = [None]
        return actions
    
    def get_transition(self, state, action):
        player, board = deepcopy(state)
        if action is not None:
            board[action] = player
            state = -player, board
            probs = [1]
            states = [state]
        else:
            probs, actions = self.adversary.policy(state)
            states = []
            for action in actions:
                new_board = board.copy()
                new_board[action] = player
                state = -player, new_board
                states.append(state)
        return probs, states

    @staticmethod
    def get_reward(state):
        _, board = state
        sums = set(board.sum(axis=0)) | set(board.sum(axis=1))
        sums.add(board.diagonal().sum())
        sums.add(np.fliplr(board).diagonal().sum())
        if 3 in sums:
            reward = 1
        elif -3 in sums:
            reward = -1
        else:
            reward = 0
        return reward
    
    def display(self, states=None, marker1='X', marker2='o', marker_size=2000, color1='b', color2='r', interval=300):
        image = 200 * np.ones((3, 3, 3)).astype(int)
        if states is not None:
            boards = [state[1] for state in states]
        else:
            boards = None
        _, board = self.state
        return display_board(image, board, boards, marker1, marker2, marker_size, color1, color2, interval)


class ConnectFour(Game):
    """Connect Four game."""

    def init_state(self):
        board = np.zeros((6, 7)).astype(int)
        state = [self.first_player, board]
        return state

    @staticmethod
    def encode(state):
        player, board = state
        if player == 1:
            return b'\x01' + board.tobytes()
        else:
            return b'\x00' + board.tobytes()
        
    @staticmethod
    def decode(state_code):
        player_code = state_code[0]
        if player_code:
            player = 1
        else:
            player = -1
        board_code = state_code[1:]
        board = np.frombuffer(board_code, dtype=int).reshape((6, 7))
        return player, board
        
    def get_actions(self, state, player=None):
        if player is None:
            player = self.player
        current_player, board = state
        if player == current_player:
            actions = np.argwhere(board[0] == 0).ravel()
        else:
            actions = [None]
        return actions

    def get_transition(self, state, action):
        player, board = deepcopy(state)
        if action is not None:
            row = 5 - np.sum(np.abs(board[:, action]))
            board[row, action] = player
            state = -player, board
            probs = [1]
            states = [state]
        else:
            try:
                probs, actions = self.adversary.policy(state)
                states = []
                for action in actions:
                    row = 5 - np.sum(np.abs(board[:, action]))
                    new_board = board.copy()
                    new_board[row, action] = player
                    new_state = -player, new_board
                    states.append(new_state)
            except: 
                print(state)
                print(action)
                print(actions)
        return probs, states
    
    @staticmethod
    def get_reward(state):
        _, board = state
        sep = ','
        sequence = np.array2string(board, separator=sep)
        sequence += np.array2string(board.T, separator=sep)
        sequence += ''.join([np.array2string(board.diagonal(offset=k), separator=sep) for k in range(-2, 4)])
        sequence += ''.join([np.array2string(np.fliplr(board).diagonal(offset=k), separator=sep) for k in range(-2, 4)])
        pattern_pos = sep.join(4 * [' 1'])
        pattern_neg = sep.join(4 * ['-1'])
        if pattern_pos in sequence:
            reward = 1
        elif pattern_neg in sequence:
            reward = -1
        else:
            reward = 0
        return reward

    def display(self, states=None, marker1='o', marker2='o', marker_size=1000, color1='gold', color2='r', interval=200):
        image = np.zeros((6, 7, 3)).astype(int)
        image[:, :, 2] = 255
        if states is not None:
            boards = [state[1] for state in states]
        else:
            boards = None
        _, board = self.state
        return display_board(image, board, boards, marker1, marker2, marker_size, color1, color2, interval)


class Nim(Game):
    """Nim game."""

    Init_State = [1, 3, 5, 7]

    @classmethod
    def set_init_state(cls, init_state):
        cls.Init_State = init_state

    def init_state(self):
        board = np.array(Nim.Init_State).astype(int)
        state = [self.first_player, board]
        return state

    @staticmethod
    def encode(state):
        player, board = state
        if player == 1:
            return b'\x01' + board.tobytes()
        else:
            return b'\x00' + board.tobytes()
        
    @staticmethod
    def decode(state_code):
        player_code = state_code[0]
        if player_code:
            player = 1
        else:
            player = -1
        board_code = state_code[1:]
        board = np.frombuffer(board_code, dtype=int).reshape((4,))
        return player, board
    
    def get_actions(self, state, player=None):
        if player is None:
            player = self.player
        current_player, board = state
        if player == current_player:
            rows = np.where(board)[0]
            actions = [(row, number + 1) for row in rows for number in range(board[row])]
        else:
            actions = [None]
        return actions

    @staticmethod
    def get_transition(state, action):
        player, board = deepcopy(state)
        row, number = action
        board[row] -= number
        state = -player, board
        probs = [1]
        states = [state]
        return probs, states

    @staticmethod
    def get_reward(state):
        player, board = state
        if np.sum(board) > 0:
            reward = 0
        else:
            reward = player
        return reward

    @staticmethod
    def is_terminal(state):
        _, board = state
        return not np.sum(board)

    def display(self, states=None, marker='d', marker_size=500, color='gold', interval=200):
        board = np.array(Nim.Init_State).astype(int)
        image = np.zeros((len(board), np.max(board), 3)).astype(int)
        image[:, :, 1] = 135
        if states is not None:
            positions = []
            for _, board in states:
                x = []
                y = []
                for row in np.where(board)[0]:
                    for col in range(board[row]):
                        x.append(row)
                        y.append(col)
                positions.append((x, y))
        else:
            positions = None
        _, board = self.state
        x = []
        y = []
        for row in np.where(board)[0]:
            for col in range(board[row]):
                x.append(row)
                y.append(col)
        position = x, y
        return display_position(image, position, positions, marker, marker_size, color, interval)
