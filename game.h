#pragma once

#include <iostream>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>

using Bitboard = uint64_t;

class ConnectFourBitboard {
private:
    Bitboard boards[2] = {0, 0};

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

    // Checks if a specific row & column position is set among any of the bitboards
    bool isSet(int row, int col) {
        Bitboard combined = boards[0] | boards[1];
        Bitboard mask = 1ULL << (row * 7) + col;
        return (mask & combined) == 0;
    }

    bool checkHorizontalWin(Bitboard board) {
        // Iterate through each row
        for (int row = 0; row < 6; ++row) {
            // Extract just the bits for this row
            Bitboard rowBits = (board >> (row * 7)) & 0x7F;
            
            // Look for four in a row within this row
            for (int shift = 0; shift <= 3; ++shift) {
                if ((rowBits & (0xF << shift)) == (0xF << shift)) {
                    return true;
                }
            }
        }
        return false;
    }

    // Checks if a connect 4 has been completed
    bool checkWin(Bitboard board) {
        // Vertical win check
        Bitboard vertical = board & (board >> 7);
        vertical &= (vertical >> 7);
        if (vertical & (vertical >> 7)) return true;

        // Horizontal win check
        if (checkHorizontalWin(board)) return true;

        // Diagonal (top-left to bottom-right) win check
        Bitboard diag1 = board & (board >> 6);
        if (diag1 & (diag1 >> 12)) return true;

        // Diagonal (top-right to bottom-left) win check
        Bitboard diag2 = board & (board >> 8);
        if (diag2 & (diag2 >> 16)) return true;

        return false;
    }

    // Checks for drawn game
    bool isBoardFull() {
        Bitboard combined = boards[0] | boards[1];
        for (int col = 0; col <= 7; ++col) {
            if (!isColumnFull(col)) {
                return false;
                break;
            }
        }
        return true;
    }


public:
    int gameState = 0;
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

    // Returns -1 if invalid move, 0 if game is still going, 1 if win condition was detected, 2 if board is full and no win detected
    void makeMove(int col) {
        if (col < 0 || col > 7) {
            return;
        }
        for (int row = 0; row <= 5; ++row) {
                Bitboard movePosition = 1ULL << (row * 7) + col;
                Bitboard combined = boards[0] | boards[1];

                if (isSet(row, col) && !isColumnFull(col)) {
                    setBit(currentPlayer, row, col);

                    if (checkWin(boards[currentPlayer])) {
                        gameState = 1;
                        return;
                    } else if (isBoardFull()) {
                        gameState = 2;
                        return;
                    }
                    currentPlayer ^= 1;

                    break;
                    return;
                } 
            }
        }
};
