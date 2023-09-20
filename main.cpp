#include <iostream>
#include <cstdint>
#include <cmath>
#include <vector>

using Bitboard = uint64_t;

class ConnectFourBitboard {
public:
    Bitboard boards[2] = {0, 0};
    int currentPlayer = 0;

    void printBoard() {
    for (int row = 5; row >= 0; --row) {
        for (int col = 0; col < 7; ++col) {
            Bitboard mask = 1ULL << (row * 7) + col;
            if (boards[0] & mask) {
                std::cout << "1 ";
            } else if (boards[1] & mask) {
                std::cout << "2 ";
            } else {
                std::cout << ". ";
            }
        }
        std::cout << "\n";
    }
    std::cout << "-----------------\n";
    }

    void setBit(int currentPlayer, int row, int col) {
        if (row > 5 || col > 6) {
            return;
        } else {
            int boardToUpdate = (currentPlayer == 0) ? 0 : 1;
            Bitboard mask = 1ULL << (row * 7) + col;
            boards[boardToUpdate] = boards[boardToUpdate] + mask;
        }
    }

    // Any column is a legal move unless the column is full
    bool isColumnFull(int col) {
        Bitboard combined = boards[0] | boards[1];
        Bitboard topOfColumn = 1ULL << (5 * 7) + col;
        // If top of column is occupied then the column is full
        return (combined & topOfColumn) != 0;
    }

};



int main() {
    ConnectFourBitboard game;
    
    for (int i = 0; i <= 5; ++i) {
        game.setBit(0, i, 1);
        game.setBit(1, i, 2);
    }

    game.printBoard();

    std::string fullColumn;
    if (game.isColumnFull(2) == true) {
        fullColumn = "Yes";
    } else {
        fullColumn = "No";
    }

    std::cout << fullColumn << std::endl;

    return 0;
}