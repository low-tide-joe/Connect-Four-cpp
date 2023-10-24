import ConnectFourBitboard as g
import numpy as np

game = g.ConnectFourBitboard()

game.makeMove(0)
game.makeMove(6)
game.makeMove(1)

game.printBoard()

adj = game.getAdjacentPositions(game.getPlayerBoardState(0), game.getPlayerBoardState(1))
print(np.log2(adj))

