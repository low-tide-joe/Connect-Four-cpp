#include <random>
#include <bitset>
#include "src/game.cpp"
#include "math.h"

bool evaluate(ConnectFourBitboard &gameToEval) {
    return (gameToEval.gameState == 1) ? true : false;
}

int minimax(ConnectFourBitboard game, int depth, bool isMaximizing, int alpha, int beta) {
    if (depth == 0 || game.gameState != 0) {
        return (evaluate(game)) ? 1 : 0;
    }

    if (isMaximizing) {
        int maxEval = INT_MIN;
        for (int move : game.getAvailableActions()) {
            ConnectFourBitboard newBoard = game;
            newBoard.makeMove(move);
            int eval = minimax(newBoard, depth -1, false, alpha, beta);
            maxEval = std::max(maxEval, eval);

            alpha = std::max(alpha, eval);
            if (beta <= alpha) 
                break;
        }
        return maxEval;
    } else {
        int minEval = INT_MAX;
        for (int move : game.getAvailableActions()) {
            ConnectFourBitboard newBoard = game;
            newBoard.makeMove(move);
            int eval = minimax(newBoard, depth -1, true, alpha, beta);
            minEval = std::min(minEval, eval);

            beta = std::min(beta, eval);
            if (beta <= alpha)
                break;
        }
        return minEval;
    }
    
}


int findBestMove(ConnectFourBitboard game, int depth) {
    int bestValue = INT_MIN;
    int bestMove;
    int alpha = INT_MIN;
    int beta = INT_MAX;

    for (int move : game.getAvailableActions()) {
        ConnectFourBitboard newBoard = game;
        newBoard.makeMove(move);
        int moveValue = minimax(newBoard, depth -1, false, alpha, beta);
        if (moveValue > bestValue) {
            bestValue = moveValue;
            bestMove = move;
        }
    }
    return bestMove;
}


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


int main() {
    ConnectFourBitboard game;
    int depth = 12;

    while (game.gameState == 0) {
        int bestMove = findBestMove(game, depth);
        std::cout << "Best Move is column: " << bestMove << "\n";
        game.makeMove(bestMove);
        if (game.gameState != 0) {
            break;
        }
        game.printBoard();
        int bestMove2 = findBestMove(game, depth);
        game.makeMove(bestMove2);
    }

    game.printBoard();
    return 0;
}