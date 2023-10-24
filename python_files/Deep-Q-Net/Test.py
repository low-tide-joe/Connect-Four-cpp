from DQN import QNetwork
import ConnectFourBitboard as g
import torch
from DQN import state_to_tensor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model_states/DQN_Model.pth"

loaded_model = QNetwork().to(device)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()


def get_best_action(model, board_state):
    player_state = board_state.getPlayerBoardState(0)
    opponent_state = board_state.getPlayerBoardState(1)
    state_tensor = state_to_tensor(player_state, opponent_state).unsqueeze(0)

    # Obtain available actions (legal moves) for the current board state
    available_actions = board_state.getAvailableActions()

    with torch.no_grad():
        q_values = model(state_tensor)
        print(q_values)
        
        # Set the Q-values of non-available actions to a large negative value
        for action in range(7):
            if action not in available_actions:
                q_values[0][action] = float('-inf')

    best_action = torch.argmax(q_values).item()
    return best_action



test_game = g.ConnectFourBitboard()
test_game.reset()


while test_game.gameState == 0:
    action = get_best_action(loaded_model, test_game)
    print(f"The best action for the given state is: {action}\n")
    test_game.makeMove(action)
    test_game.printBoard()
    player_move = int(input("Your move: "))
    test_game.makeMove(player_move)
