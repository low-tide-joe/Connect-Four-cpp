from DQN import QNetwork
from DQN import state_to_tensor
from DQN import random_policy
import ConnectFourBitboard as g
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model_states/DQN_Model.pth"

loaded_model = QNetwork().to(device)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()


def get_best_action(model, board_state, print_q_values=True):
    player_state = board_state.getPlayerBoardState(0)
    opponent_state = board_state.getPlayerBoardState(1)
    state_tensor = state_to_tensor(player_state, opponent_state).unsqueeze(0)

    # Obtain available actions (legal moves) for the current board state
    available_actions = board_state.getAvailableActions()

    with torch.no_grad():
        q_values = model(state_tensor)
        if print_q_values == True: print(q_values)
        
        # Set the Q-values of non-available actions to a large negative value
        for action in range(7):
            if action not in available_actions:
                q_values[0][action] = float('-inf')

    best_action = torch.argmax(q_values).item()
    return best_action


def test_vs_human():
    test_game = g.ConnectFourBitboard()
    test_game.reset()

    while test_game.gameState == 0:
        action = get_best_action(loaded_model, test_game, print_q_values=True)
        print(f"The best action for the given state is: {action}\n")
        test_game.makeMove(action)
        test_game.printBoard()
        player_move = int(input("Your move: "))
        while player_move not in test_game.getAvailableActions():
            player_move = int(input("Invalid Move, try again: "))
        test_game.makeMove(player_move)


def test_vs_random(num_games=100):
    test_game = g.ConnectFourBitboard()
    test_game.reset()
    agent_wins = 0
    random_wins = 0
    draws = 0

    for i in range(num_games):
        test_game.reset()
        while test_game.gameState == 0:
            agent_action = get_best_action(loaded_model, test_game, print_q_values=False)
            test_game.makeMove(agent_action)
            random_action = random_policy(test_game.getAvailableActions())
            test_game.makeMove(random_action)
            
        if test_game.currentPlayer == 0:
            agent_wins += 1
        elif test_game.currentPlayer == 1:
            random_wins += 1
        elif test_game.gameState == 2:
            draws += 1

    print(f"We played {num_games} games.\nAI Agent wins: {agent_wins}\nRandom Agent wins: {random_wins}\nDraws: {draws}\nAgent Win Rate = {agent_wins / (agent_wins + random_wins)}")
        


# test_vs_human()
test_vs_random(num_games=1000)


