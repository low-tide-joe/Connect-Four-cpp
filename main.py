import ConnectFourBitboard as g
import numpy as np
import random 
import QLearnAgent

game = g.ConnectFourBitboard()

def get_reward(game):
    if game.gameState == 0:
        return 0
    elif game.gameState == 1:
        if game.currentPlayer == 0:
            return 1 # Player 1 wins
        else:
            return -1 # Player 2 wins
    elif game.gameState == 2: # draw game
        return 0


agent = QLearnAgent.QLearningAgent()

num_episodes = 10000

for i in range(num_episodes):
    game.reset()
    while game.gameState == 0:
        player1_bitboard = game.getPlayerBoardState(0)
        player2_bitboard = game.getPlayerBoardState(1)
        state_key = (player1_bitboard, player2_bitboard) 

        available_actions = game.getAvailableActions()
        action = agent.select_action(state_key, available_actions)

        game.makeMove(action)

        reward = get_reward(game)

        next_player1_bitboard = game.getPlayerBoardState(0)
        next_player2_bitboard = game.getPlayerBoardState(1)
        next_state_key = (next_player1_bitboard, next_player2_bitboard)

        agent.update_q_value(state_key, action, reward, next_state_key)
        