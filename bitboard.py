"""Bitboard class for Connect 4."""
import random


class BitBoard:
    """Board class for Connect 4."""
    height: int
    width: int
    winning_length: int
    move_history: list[int]
    bitboard: list[int]
    __counter: int
    __heights: list[int]  # current height
    zobrist_table: list[list[list[int]]]

    def __init__(self, height: int = 6, width: int = 7, winning_length: int = 4) -> None:
        self.height = height
        self.width = width
        self.winning_length = winning_length
        self.move_history = []
        self.__counter = 0
        self.bitboard = [0] * 2  # We always have two players
        # positions of the next possible move in each column
        self.__heights = [i * (height + 1) for i in range(width)]
        self.shifts = [1, self.height, self.height + 1, self.height + 2]
        # Initialize Zobrist table: [player(0/1)][col][row]
        self.zobrist_table = [[[random.getrandbits(64) for _ in range(
            height)] for _ in range(width)] for _ in range(2)]
        self.current_hash = 0
        assert len(self.__heights) == width

    def make_move(self, col: int) -> None:
        """Make a move in the column."""
        # assert 0 <= col < self.width
        row = self.__heights[col] - col * (self.height + 1)
        player = self.turn()
        # Update bitboard
        _move: int = 1 << self.__heights[col]
        self.bitboard[player] ^= _move
        # Update Zobrist hash
        self.current_hash ^= self.zobrist_table[player][col][row]
        self.__heights[col] += 1
        self.move_history.append(col)
        self.__counter += 1

    def undo_move(self) -> None:
        """Undo the last move."""
        col = self.move_history.pop()
        self.__counter -= 1
        self.__heights[col] -= 1
        # Determine row index after undo (the position we remove)
        row = self.__heights[col] - col * (self.height + 1)
        player = self.turn()
        _move: int = 1 << self.__heights[col]
        self.bitboard[player] ^= _move

        # Update Zobrist hash
        self.current_hash ^= self.zobrist_table[player][col][row]

    def is_winning(self, bitboard: int | None = None) -> bool:
        """Check if the passed board is winning or the current player has won."""
        if bitboard is None:
            bitboard = self.bitboard[self.turn() ^ 1]
        # corresponding to vertical, diagonal \, horizontal, diagonal /
        for shift_length in self.shifts:
            copy_bitboard = bitboard
            for i in range(1, self.winning_length):
                copy_bitboard &= (bitboard >> shift_length * i)
                if copy_bitboard == 0:
                    break
            if copy_bitboard != 0:
                return True
        return False

    def list_moves(self) -> list[int]:
        """List all possible moves."""
        moves = []
        for col in range(self.width):
            if not self.__top_mask(col) & (1 << self.__heights[col]):
                moves.append(col)
        return moves

    def is_terminal(self) -> tuple[bool, int]:
        """Check if the board is terminal and return the last move's player."""
        return self.is_winning() or \
            self.__counter == self.height * \
            self.width, (self.turn() ^ 1)  # self.turn()?

    def turn(self) -> int:
        """Return the current player."""
        return self.__counter & 1

    def __top_mask(self, col: int) -> int:
        """Return the top mask for the column."""
        return (1 << self.height) << (col * (self.height + 1))

    def __str__(self) -> str:
        """Return the string representation of the board."""
        result = []
        for i in range(self.height - 1, -1, -1):
            row = []
            for j in range(self.width):
                if self.bitboard[0] & (1 << (j * (self.height + 1) + i)):
                    row.append("X")
                elif self.bitboard[1] & (1 << (j * (self.height + 1) + i)):
                    row.append("O")
                else:
                    row.append(".")
            result.append(f"{i+1:02d} " + "  ".join(row))
        return "\n".join(result) + "\n  " + " ".join([f"{i+1:02d}" for i in range(self.width)])
