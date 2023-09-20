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

    while (game.gameState == 0) {
        randomMove = chooseRandomColumn();
        game.makeMove(randomMove);
    }

    game.printBoard();

    std::string endGameMessage = (game.gameState == 1) ? "Player " + std::to_string(game.currentPlayer + 1) + " Wins!" : "Draw Game.";
    std::cout << endGameMessage << "\nLast move played was in column " << std::to_string(randomMove) << " by player: " << std::to_string(game.currentPlayer + 1) << std::endl;

    return 0;
}