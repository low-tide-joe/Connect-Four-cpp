import ConnectFourBitboard as g
import random as rnd


game = g.ConnectFourBitboard()

player1wins = 0
player2wins = 0
draws = 0

for i in range(0, 10000):
    while (game.gameState == 0):
        randomColumn = rnd.randint(0, 6)
        game.makeMove(randomColumn)
    
    if (game.gameState == 1):
        if game.currentPlayer == 0:
            player1wins += 1
        else:
            player2wins += 1
    elif game.gameState == 2:
        draws += 1
    
    game.reset()


print("Player 1 wins: " + str(player1wins) + "\nPlayer 2 wins: " + str(player2wins) + "\nDraws: " + str(draws))

