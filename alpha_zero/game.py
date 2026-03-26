import numpy as np

ROWS = 6
COLS = 7


class ConnectFour:
    """
    Connect Four game engine.

    Board is a (6, 7) numpy array:
        1  = player 1's piece
       -1  = player 2's piece
        0  = empty

    Row 0 is the bottom row (gravity drops pieces to the lowest empty row).
    current_player is always +1 or -1.

    make_move() returns a NEW ConnectFour instance — states are immutable,
    which avoids shared-state bugs in the MCTS tree.
    """

    def __init__(self):
        self.board = np.zeros((ROWS, COLS), dtype=np.int8)
        self.current_player = 1
        self._winner = None      # cached: 1, -1, or 0 (draw)
        self._terminal = False
        self.last_move = None

    # ------------------------------------------------------------------
    # Core game logic
    # ------------------------------------------------------------------

    def get_valid_moves(self) -> np.ndarray:
        """Returns a boolean array of length COLS. True = column not full."""
        return self.board[ROWS - 1] == 0

    def make_move(self, col: int) -> "ConnectFour":
        """
        Drop a piece in `col` for the current player.
        Returns a new ConnectFour instance with the move applied.
        Raises ValueError if the column is full or out of range.
        """
        if col < 0 or col >= COLS:
            raise ValueError(f"Column {col} is out of range.")
        if self.board[ROWS - 1, col] != 0:
            raise ValueError(f"Column {col} is full.")

        next_state = ConnectFour.__new__(ConnectFour)
        next_state.board = self.board.copy()
        next_state.current_player = -self.current_player
        next_state._winner = None
        next_state._terminal = False
        next_state.last_move = col

        # Find the lowest empty row in this column (row 0 = bottom)
        row = 0
        while row < ROWS and next_state.board[row, col] != 0:
            row += 1

        next_state.board[row, col] = self.current_player
        next_state._check_terminal(row, col, self.current_player)
        return next_state

    def is_terminal(self) -> bool:
        return self._terminal

    def get_result(self, perspective: int) -> float:
        """
        Returns the game result from `perspective`'s point of view.
        +1 if `perspective` won, -1 if lost, 0 if draw.
        Returns None if the game is not over.
        """
        if not self._terminal:
            return None
        if self._winner == 0:
            return 0.0
        return 1.0 if self._winner == perspective else -1.0

    def get_canonical_board(self) -> np.ndarray:
        """
        Returns the board from the current player's perspective:
        current player's pieces = +1, opponent's pieces = -1.
        This lets the neural network always reason as "my pieces vs theirs".
        """
        return self.board * self.current_player

    # ------------------------------------------------------------------
    # Terminal detection
    # ------------------------------------------------------------------

    def _check_terminal(self, row: int, col: int, player: int):
        """Check if the move at (row, col) by `player` ended the game."""
        if self._check_win(row, col, player):
            self._winner = player
            self._terminal = True
        elif not self.get_valid_moves().any():
            self._winner = 0  # draw
            self._terminal = True

    def _check_win(self, row: int, col: int, player: int) -> bool:
        """
        Check if `player` has four in a row after placing at (row, col).
        Checks all four directions through the placed piece.
        """
        b = self.board
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horiz, vert, diag/, diag\
        for dr, dc in directions:
            count = 1
            # Extend in positive direction
            r, c = row + dr, col + dc
            while 0 <= r < ROWS and 0 <= c < COLS and b[r, c] == player:
                count += 1
                r += dr
                c += dc
            # Extend in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < ROWS and 0 <= c < COLS and b[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            if count >= 4:
                return True
        return False

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        symbols = {1: "X", -1: "O", 0: "."}
        rows = []
        for row in range(ROWS - 1, -1, -1):  # top to bottom
            rows.append(" ".join(symbols[self.board[row, c]] for c in range(COLS)))
        rows.append(" ".join(str(c) for c in range(COLS)))
        return "\n".join(rows)


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Test horizontal win
    g = ConnectFour()
    for col in [0, 0, 1, 1, 2, 2, 3]:
        g = g.make_move(col)
    assert g.is_terminal(), "Should be terminal after horizontal win"
    assert g.get_result(1) == 1.0, "Player 1 should win"
    assert g.get_result(-1) == -1.0, "Player 2 should lose"
    print("Horizontal win: PASS")

    # Test vertical win
    g = ConnectFour()
    for col in [0, 1, 0, 1, 0, 1, 0]:
        g = g.make_move(col)
    assert g.is_terminal()
    assert g.get_result(1) == 1.0
    print("Vertical win: PASS")

    # Test diagonal win (/)
    g = ConnectFour()
    moves = [0, 1, 1, 2, 3, 2, 2, 3, 3, 5, 3]
    for col in moves:
        g = g.make_move(col)
    assert g.is_terminal()
    assert g.get_result(1) == 1.0
    print("Diagonal (/) win: PASS")

    # Test column overflow
    g = ConnectFour()
    for _ in range(6):
        g = g.make_move(0) if g.current_player == 1 else g.make_move(1)
        if g.is_terminal():
            break
    try:
        # Fill a column completely then try again
        g2 = ConnectFour()
        for i in range(6):
            g2 = g2.make_move(0)
            if g2.is_terminal():
                break
        if not g2.is_terminal():
            g2.make_move(0)
            assert False, "Should have raised ValueError"
    except ValueError:
        print("Column overflow: PASS")

    # Test canonical board
    g = ConnectFour()
    g = g.make_move(3)  # player 1 moves
    canon = g.get_canonical_board()
    # Now it's player -1's turn; canonical should flip signs
    assert canon[0, 3] == -1, "Player 1's piece should appear as -1 from player -1's view"
    print("Canonical board: PASS")

    print("\nAll tests passed.")
