#include <random>
#include "game.h"

int chooseRandomColumn() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 7);

    return distrib(gen);
}


int main() {
    ConnectFourBitboard game;
    int randomMove;
    int player1WinCount = 0;
    int player2WinCount = 0;
    int draws = 0;

    for (int i = 0; i <= 1000000; ++i) {
        while (game.gameState == 0) {
            randomMove = chooseRandomColumn();
            game.makeMove(randomMove);
        }

        if (game.gameState == 1) {
            (game.currentPlayer == 0) ? player1WinCount++ : player2WinCount++;
        } else if (game.gameState == 2) {
            draws++;
        }
        //std::cout << "Game " << i << " completed, Game state = " << game.gameState << std::endl;
        game.reset();
    }

std::cout << "Player 1 Wins: " << player1WinCount << "\nPlayer 2 Wins: " << player2WinCount << "\nDraws: " << draws << std::endl;

/*
    while (game.gameState == 0) {
        randomMove = chooseRandomColumn();
        game.makeMove(randomMove);
    }

    game.printBoard();

    std::string endGameMessage = (game.gameState == 1) ? "Player " + std::to_string(game.currentPlayer + 1) + " Wins!" : "Draw Game.";
    std::cout << endGameMessage << "\nLast move played was in column " << std::to_string(randomMove) << " by player: " << std::to_string(game.currentPlayer + 1) << std::endl;
*/
    return 0;
}