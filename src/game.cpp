#include "game.hpp"


void ConnectFourBitboard::setBit(int currentPlayer, int row, int col) {
    if (row > 5 || col > 6) {
        return;
    } else {
        int boardToUpdate = (currentPlayer == 0) ? 0 : 1;
        Bitboard mask = 1ULL << (row * 7) + col;
        boards[boardToUpdate] |= mask;
    }
}


bool ConnectFourBitboard::isColumnFull(int col) {
    Bitboard combined = boards[0] | boards[1];
    Bitboard topOfColumn = 1ULL << (5 * 7) + col;
    return (combined & topOfColumn) != 0;
}


bool ConnectFourBitboard::isSet(int row, int col) {
    Bitboard combined = boards[0] | boards[1];
    Bitboard mask = 1ULL << (row * 7) + col;
    return (mask & combined) == 0;
}


bool ConnectFourBitboard::checkVerticalWin(Bitboard board) {
    for (int col = 0; col < 7; ++col) {
        Bitboard colBits = 0;
        for (int row = 0; row < 6; ++row) {
            Bitboard position = (row * 7) + col;
            colBits |= ((board & (1ULL << position)) != 0) << row;
        }

        for (int shift = 0; shift <= 2; ++shift) {
            if ((colBits & (0xF << shift)) == (0xF << shift)) {
                return true;
            }
        }
    }
    return false;
}


bool ConnectFourBitboard::checkHorizontalWin(Bitboard board) {
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


bool ConnectFourBitboard::checkDiagonalWin(Bitboard board) {
        for (int row = 0; row < 6; ++row) {
            for (int col = 0; col < 7; ++col) {
                Bitboard position = (row * 7) + col;
                // Upward diagonal
                if (row <= 2 && col <= 3) {
                    if (
                        ((board & (1ULL << position)) != 0) &&
                        ((board & (1ULL << (position + 8))) != 0) &&
                        ((board & (1ULL << (position + 16))) != 0) &&
                        ((board & (1ULL << (position + 24))) != 0)
                    ) {
                        return true;
                    }
                }

                if (row >= 3 && col <= 3) {
                    if (
                        ((board & (1ULL << position)) != 0) &&
                        ((board & (1ULL << (position - 6))) != 0) &&
                        ((board & (1ULL << (position - 12))) != 0) &&
                        ((board & (1ULL << (position - 18))) != 0) 
                    ) {
                        return true;
                    }
                }
            }
        }
        return false;    
}


bool ConnectFourBitboard::checkWin(Bitboard board) {
    // Vertical win check
    if (checkVerticalWin(board)) {
        //std::cout << "Vertical Win" << std::endl;
        return true;
    }

    // Horizontal win check
    if (checkHorizontalWin(board)) {
        //std::cout << "Horizontal Win" << std::endl;
        return true;
    }

    // Diagonal win check
    if (checkDiagonalWin(board)) {
        //std::cout << "Diagonal Win" << std::endl;
        return true;
    }

    return false;    
}


bool ConnectFourBitboard::isBoardFull() {
    Bitboard combined = boards[0] | boards[1];
    for (int col = 0; col < 7; ++col) {
        if (!isColumnFull(col)) {
            return false;
        }
    }
    return true;
}


void ConnectFourBitboard::printBoard() {
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


void ConnectFourBitboard::makeMove(int col) {
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


Bitboard ConnectFourBitboard::getPlayerBoardState(int player) {
    if (player < 0 || player >= 2) throw std::out_of_range("Invalid Player");
    return boards[player];
}


const std::vector<int>& ConnectFourBitboard::getAvailableActions() {
    availableActions.clear();
    availableActions.reserve(7);
    for (int col = 0; col < 7; ++col) {
        if (!isColumnFull(col)) {
            availableActions.push_back(col);
        }
    }
    return availableActions;
}


void ConnectFourBitboard::reset() {
    boards[0] = 0;
    boards[1] = 0;
    currentPlayer = 0;
    gameState = 0;
}


bool ConnectFourBitboard::isAccessible(Bitboard board, Bitboard position) {
    if (board & position) return false;
    if (position & 0x7F) return true;
    if (board & (position >> 7)) return true;
    return false;
}


std::vector<Bitboard> ConnectFourBitboard::getAdjacentPositions(int currentPlayer) {
    std::vector<Bitboard> adjacencies;
    const Bitboard Left_Column_Mask = 0x810204081;
    const Bitboard Right_Column_Mask = 0x20408102040;
    Bitboard player = getPlayerBoardState(currentPlayer);
    Bitboard opponent = getPlayerBoardState(1 - currentPlayer);

    Bitboard left = ((player & ~Left_Column_Mask) >> 1) & ~(player | opponent);
    Bitboard right = ((player & ~Right_Column_Mask) << 1) & ~(player | opponent);
    Bitboard up = (player << 7) & ~(player | opponent);
    Bitboard down = (player >> 7) & ~(player | opponent);
    Bitboard upRight = (up << 1) & ~Left_Column_Mask;
    Bitboard upLeft = (up >> 1) & ~Right_Column_Mask;
    Bitboard downRight = (down << 1) & ~Left_Column_Mask;
    Bitboard downLeft = (down >> 1) & ~Right_Column_Mask;

    Bitboard combined = left | right | up | down | upRight | upLeft | downRight | downLeft;
    for (Bitboard pos = 1; pos != 0; pos <<= 1) {
        if (combined & pos) {
            if (isAccessible(player | opponent, pos)) {
                adjacencies.push_back(pos);
            }
        }
    }

    return adjacencies;

}