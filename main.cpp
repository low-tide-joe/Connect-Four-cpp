#include <random>
#include <bitset>
#include "src/game.cpp"
#include "math.h"

int chooseRandomColumn(const std::vector<int> &availableActions) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, availableActions.size() - 1);

    if (availableActions.empty()) {
        std::cerr << "Vector empty!\n";
        return -1;
    }
    
    int random_index = distr(gen);
    int random_element = availableActions[random_index];

    return random_element;
}


void testAdjacentSquaresFunction() {
    ConnectFourBitboard game;

    game.makeMove(6);

    game.printBoard();

    std::vector<Bitboard> adjacent = game.getAdjacentPositions(0);

    for (int Bitboard : adjacent) {
        std::cout << log2(Bitboard) << "\n";
    }    
}


int main() {

    testAdjacentSquaresFunction();
    return 0;

    ConnectFourBitboard game;
    int randomMove;
    int player1WinCount = 0;
    int player2WinCount = 0;
    int draws = 0;


    for (int i = 0; i < 1000; ++i) {
        while (game.gameState == 0) {
            std::vector<int> availableActions = game.getAvailableActions();
            randomMove = chooseRandomColumn(availableActions);
            if (randomMove == -1) {
                break;
            }
            game.makeMove(randomMove);
        }

        if (game.gameState == 1) {
            (game.currentPlayer == 0) ? player1WinCount++ : player2WinCount++;
        } else if (game.gameState == 2) {
            draws++;
        }
        game.reset();
    }

    std::cout << "Player 1 Wins: " << player1WinCount << "\nPlayer 2 Wins: " << player2WinCount << "\nDraws: " << draws << std::endl;

    return 0;
}