import ConnectFourBitboard as g
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        
        # Convolutional Layers
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 6 * 7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)

        # Dropout layer
        self.dropout1 = nn.Dropout(0.4)       
        self.dropout2 = nn.Dropout(0.3)

        # Batch Normalisation 
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)

        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        return self.fc3(x)


def bitboard_to_tensor(bitboard):
    binary_str = format(bitboard, '042b')
    binary_list = list(map(int, list(binary_str)))
    binary_tensor = torch.tensor(binary_list, dtype=torch.float32).view(6, 7)
    return binary_tensor


def state_to_tensor(player_state, opponent_state):
    player_tensor = bitboard_to_tensor(player_state)
    opponent_tensor = bitboard_to_tensor(opponent_state)
    return torch.stack([player_tensor, opponent_tensor])


def compute_rewards(gameState, currentPlayer, adjacency_difference, connection, win_reward=1, adjacency_bonus=0.2, connection_bonus=0.6, small_penalty=-0.05):
    if gameState == 0:
        if connection == False: connection_bonus = 0
        return (adjacency_bonus * adjacency_difference) + connection_bonus + small_penalty
    elif gameState == 1:
        return 0 #win_reward
    elif gameState == 2:
        return 0
    

def get_adjacent_action_column(adjacencies):
    adjacent_actions = []
    for i in adjacencies:
        column = np.log2(i) % 7
        adjacent_actions.append(column)
    # returns adjacent actions as a sorted list of integers
    return sorted([int(x) for x in adjacent_actions])
    
    
def epsilon_greedy_policy(agent, state, epsilon, available_actions):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(available_actions)
    else:
        with torch.no_grad():
            q_values = agent(state)
            # set the Q-values of non-available actions to a large negative value
            for action in range(7):
                if action not in available_actions:
                    q_values[0][action] = float('-inf')
            return torch.argmax(q_values).item()
        

def random_policy(available_actions):
    return np.random.choice(available_actions)
        

def train(episodes=1000, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99, current_agent=0, previous_agent=0, criterion=0, optimizer=0, agent_update_frequency=100):
    epsilon = epsilon_start
    for episode in range(episodes):
        game = g.ConnectFourBitboard()
        game.reset()

        while game.gameState == 0:
            current_player = game.currentPlayer
            opponent = 1 - current_player

            player_state = game.getPlayerBoardState(current_player)
            opponent_state = game.getPlayerBoardState(opponent)
            state = state_to_tensor(player_state, opponent_state).unsqueeze(0)

            available_actions = game.getAvailableActions()

            # Choose action using the current_agent.
            action = epsilon_greedy_policy(current_agent, state, epsilon, available_actions)

            # Get the list of adjacencies for the pre-move board state
            adjacencies_pre = game.getAdjacentPositions(current_player)

            # Get the list of actions that would result in creating a conneciton 
            actions_resulting_in_connection = get_adjacent_action_column(adjacencies_pre)
            if action in actions_resulting_in_connection:
                connection = True
            else:
                connection = False

            game.makeMove(action)

            # Get the list of adjacencies post-move and calculate the difference
            adjacencies_post = game.getAdjacentPositions(current_player)
            adjacency_difference = len(adjacencies_post) - len(adjacencies_pre)

            player_state_next = game.getPlayerBoardState(current_player)
            opponent_state_next = game.getPlayerBoardState(opponent)
            next_state = state_to_tensor(player_state_next, opponent_state_next).unsqueeze(0)
                        
            reward = compute_rewards(game.gameState, current_player, adjacency_difference, connection)

            done = game.gameState != 0

            # Get target Q-value from previous_agent.
            with torch.no_grad():
                target = reward + gamma * torch.max(previous_agent(next_state)) * (not done)

            # Update Q-value of current_agent.
            prediction = current_agent(state)[0][action]
            loss = criterion(prediction, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done:
                break

        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if (episode % 10) == 0 or (episode == episodes - 1):
            print("Game number " + str(episode) + " reached")
            game.printBoard()

        # Occasionally update the "previous" agent to be the "current" agent.
        if episode % agent_update_frequency == 0:
            previous_agent.load_state_dict(current_agent.state_dict())
            previous_agent.eval()
