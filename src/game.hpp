#pragma once

#include <iostream> 
#include <cstdint>
#include <vector>

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
  std::vector<int> availableActions;
  bool isAccessible(Bitboard board, Bitboard position);

public:
  int gameState = 0;
  int currentPlayer = 0;
  void printBoard();
  void makeMove(int col);
  void reset();  
  Bitboard getPlayerBoardState(int player);
  const std::vector<int>& getAvailableActions();
  std::vector<Bitboard> getAdjacentPositions(int currentPlayer);
  
};