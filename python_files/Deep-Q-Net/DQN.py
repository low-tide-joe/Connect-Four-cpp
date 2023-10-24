import ConnectFourBitboard as g
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random 


class QNetwork(nn.Module):
    def __init__(self):
        fc_layer_size = 256
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(2, fc_layer_size, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(fc_layer_size * 6 * 7, fc_layer_size)
        self.fc2 = nn.Linear(fc_layer_size, 7)        
    
    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def bitboard_to_tensor(bitboard):
    binary_str = format(bitboard, '042b')
    binary_list = list(map(int, list(binary_str)))
    binary_tensor = torch.tensor(binary_list, dtype=torch.float32).view(6, 7)
    return binary_tensor


def state_to_tensor(player_state, opponent_state):
    player_tensor = bitboard_to_tensor(player_state)
    opponent_tensor = bitboard_to_tensor(opponent_state)
    return torch.stack([player_tensor, opponent_tensor])


def compute_rewards(gameState, currentPlayer, adjacency_difference, adjacency_bonus=0.2):
    if gameState == 0:
        return (adjacency_bonus * adjacency_difference) 
    elif gameState == 1:
        return 1 if currentPlayer == 0 else -1
    elif gameState == 2:
        return 0
    
    
def epsilon_greedy_policy(agent, state, epsilon, available_actions):
    if random.uniform(0, 1) < epsilon:
        return random.choice(available_actions)
    else:
        with torch.no_grad():
            q_values = agent(state)
            # set the Q-values of non-available actions to a large negative value
            for action in range(7):
                if action not in available_actions:
                    q_values[0][action] = float('-inf')
            return torch.argmax(q_values).item()
        

def train(episodes=1000, gamma=0.90, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99, current_agent=0, previous_agent=0, criterion=0, optimizer=0, agent_update_frequency=100):
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

            adjacencies_pre = game.getAdjacentPositions(current_player)
            game.makeMove(action)
            adjacencies_post = game.getAdjacentPositions(current_player)
            adjacency_difference = len(adjacencies_post) - len(adjacencies_pre)

            player_state_next = game.getPlayerBoardState(current_player)
            opponent_state_next = game.getPlayerBoardState(opponent)
            next_state = state_to_tensor(player_state_next, opponent_state_next).unsqueeze(0)
                        
            reward = compute_rewards(game.gameState, current_player, adjacency_difference)

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

        if (episode % 10) == 0:
            print("Game number " + str(episode) + " reached")
            game.printBoard()

        # Occasionally update the "previous" agent to be the "current" agent.
        if episode % agent_update_frequency == 0:
            previous_agent.load_state_dict(current_agent.state_dict())
