import ConnectFourBitboard as g
import torch
import numpy as np

def bitboard_to_array(bitboard1, bitboard2):
    board = np.zeros((6, 7), dtype=int)

    for row in range(6):
        for col in range(7):
            index = row * 7 + col
            if (bitboard1 >> index) & 1:
                board[5-row, col] = 1
            elif (bitboard2 >> index) & 1:
                board[5-row, col] = -1
            
    return board


class Node:
    def __init__(self, prior, turn, state):
        self.prior = prior
        self. turn = turn
        self.state = state
        self.children = {}
        self.value = 0

    def expand(self, action_probs):
        for action, prob in enumerate(action_probs):
            if prob > 0:
                self.children[action] = Node(prior=prob, turn=(1 - self.turn), state=)


game = g.ConnectFourBitboard()
board = bitboard_to_array(game.getPlayerBoardState(0), game.getPlayerBoardState(1))

root = Node(prior=0, turn=0, state=board)