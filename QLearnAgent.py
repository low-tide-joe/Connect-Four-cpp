import numpy as np
import random

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0)
    
    def update_q_value(self, state_key, action, reward, next_state_key):
        max_q_next_state = max([self.get_q_value(next_state_key, a) for a in range(7)])
        old_q_value = self.get_q_value(state_key, action)
        self.q_table.setdefault(state_key, {})[action] = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * max_q_next_state)

    def select_action(self, state_key, available_actions):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        else:
            q_values = [self.get_q_value(state_key, action) for action in available_actions]
            return available_actions[np.argmax(q_values)]
        
