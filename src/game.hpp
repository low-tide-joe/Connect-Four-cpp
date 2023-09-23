#pragma once

#include <iostream> 
#include <cstdint>

using Bitboard = uint64_t;

class ConnectFourBitboard {
private:
  Bitboard boards[2] = {0, 0};
  void setBit(int currentPlayer, int row, int col);
  bool isColumnFull(int col);
  bool isSet(int row, int col);
  bool checkVerticalWin(Bitboard board);
  bool checkHorizontalWin(Bitboard board);
  bool checkDiagonalWin(Bitboard board);
  bool checkWin(Bitboard board);
  bool isBoardFull();

public:
  int gameState = 0;
  int currentPlayer = 0;
  void printBoard();
  void makeMove(int col);
  void reset();  
  
};