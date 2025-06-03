"""Connect 4 AI using Minmax with alpha-beta pruning."""
from sys import maxsize
import cProfile
import pstats
import random
import time
import asyncio
from typing import AsyncGenerator

INT_MAX = maxsize
INT_MIN = -maxsize


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


class Game:
    """Game class for Connect 4."""
    n_rows: int
    n_columns: int
    winning_length: int
    board: list[list[int]]
    move_history: list[tuple[int, int]]
    current_player: int
    bitboard: BitBoard

    def __init__(self, n_rows=7, n_columns=7, winning_length=4, current_player=0) -> None:
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.winning_length = winning_length
        self.current_player = current_player
        self.board = [[-1 for _ in range(n_columns)] for _ in range(n_rows)]
        self.move_history = []
        self.bitboard = BitBoard(
            height=self.n_rows, width=self.n_columns, winning_length=self.winning_length)

        H = self.n_rows
        W = self.n_columns
        K = self.winning_length
        shifts = self.bitboard.shifts
        self.line_masks: list[int] = []
        for r_start in range(H):
            for c_start in range(W):
                for shift in shifts:
                    line_mask = 0
                    valid = True
                    for i in range(K):
                        if shift == 1:
                            bit_r, bit_c = r_start + i, c_start
                        elif shift == H + 1:
                            bit_r, bit_c = r_start, c_start + i
                        elif shift == H + 2:
                            bit_r, bit_c = r_start + i, c_start + i
                        elif shift == H:
                            bit_r, bit_c = r_start - i, c_start + i
                        else:
                            valid = False
                            break

                        if not (0 <= bit_r < H and 0 <= bit_c < W):
                            valid = False
                            break

                        bit_pos = bit_c * (H + 1) + bit_r
                        line_mask |= (1 << bit_pos)

                    if not valid:
                        continue

                    self.line_masks.append(line_mask)

    def _count_set_bits(self, n: int) -> int:
        """Counts the number of set bits in an integer."""
        if hasattr(n, "bit_count"):
            return n.bit_count()
        count = 0
        while n > 0:
            n &= (n - 1)
            count += 1
        return count

    def eval(self) -> int:
        """Przyspieszona wersja funkcji oceniającej przy użyciu wcześniej obliczonych masek."""
        player0_bb = self.bitboard.bitboard[0]
        player1_bb = self.bitboard.bitboard[1]

        K = self.winning_length

        p0_potential_lines = 0
        p1_potential_lines = 0
        player0_threats = 0
        player1_threats = 0

        for line_mask in self.line_masks:
            x_bits = (line_mask & player0_bb).bit_count()
            o_bits = (line_mask & player1_bb).bit_count()
            empty = K - x_bits - o_bits

            if x_bits > 0 and o_bits > 0:
                continue

            if o_bits == 0:
                p0_potential_lines += 1
                if x_bits == K - 1 and empty == 1:
                    player0_threats += 3
                elif x_bits == K - 2 and empty == 2:
                    player0_threats += 2
                elif K >= 3 and x_bits == K - 3 and empty == 3:
                    player0_threats += 1

            if x_bits == 0:
                p1_potential_lines += 1
                if o_bits == K - 1 and empty == 1:
                    player1_threats += 3
                elif o_bits == K - 2 and empty == 2:
                    player1_threats += 2
                elif K >= 3 and o_bits == K - 3 and empty == 3:
                    player1_threats += 1

        score = (p0_potential_lines - p1_potential_lines) + \
            2 * (player1_threats - player0_threats)
        return score


class Player:
    """Player class for Connect 4."""
    __explored_nodes: int = 0

    def __init__(self, max_depth: int = 5, time_limit: float = 300.0) -> None:
        self.max_depth = max_depth
        self.time_limit = time_limit
        # Transposition Table maps hash -> (value, depth, flag, best_move)
        # flag ∈ {"EXACT", "LOWER", "UPPER"}
        self.transposition_table: dict[int,
                                       tuple[int, int, str, int | None]] = {}

    @staticmethod
    def _order_moves_base(moves: list[int], cols: int) -> list[int]:
        """Base ordering by proximity to center column."""
        center = cols // 2
        return sorted(moves, key=lambda x: abs(x - center))

    def order_moves(self, moves: list[int], cols: int, stored_move: int | None) -> list[int]:
        """Order moves, placing stored_move first if available, then by proximity to center."""
        if stored_move is not None and stored_move in moves:
            remaining = [m for m in moves if m != stored_move]
            ordered_remaining = self._order_moves_base(remaining, cols)
            return [stored_move] + ordered_remaining
        return self._order_moves_base(moves, cols)

    async def min_max(self, _game: Game, depth: int, alpha: int, beta: int, maximizing: bool,
                      start_time: float, time_limit: float) -> tuple[int, int | None]:
        """Minimax algorithm with alpha-beta pruning."""
        if time.time() - start_time > time_limit:
            raise asyncio.CancelledError()

        self.__explored_nodes += 1
        current_hash = _game.bitboard.current_hash
        # Check transposition table
        if current_hash in self.transposition_table:
            stored_value, stored_depth, stored_flag, stored_move = self.transposition_table[
                current_hash]
            if stored_depth >= depth:
                if stored_flag == "EXACT":
                    return stored_value, stored_move
                if stored_flag == "LOWER" and maximizing and stored_value >= beta:
                    return stored_value, stored_move
                if stored_flag == "UPPER" and not maximizing and stored_value <= alpha:
                    return stored_value, stored_move
        else:
            stored_move = None

        _is_terminal, _turn = _game.bitboard.is_terminal()
        if depth == 0 or _is_terminal:
            if _is_terminal:
                if _turn == 0:
                    return INT_MAX-1, None
                if _turn == 1:
                    return INT_MIN+1, None
                return 0, None
            score = _game.eval()
            # Store in transposition table
            self.transposition_table[current_hash] = (
                score, depth, "EXACT", None)
            return score, None
        raw_moves = _game.bitboard.list_moves()
        moves = self.order_moves(raw_moves, _game.n_columns, stored_move)
        best_move = None
        if maximizing:
            value = INT_MIN
            flag: str = "UPPER"  # Assume upper-bound until proven otherwise
            for move in moves:
                _game.bitboard.make_move(move)
                # Make sure to undo the move even if the function raises an error
                try:
                    new_value, _ = await self.min_max(
                        _game, depth - 1, alpha, beta, False, start_time, time_limit)
                finally:
                    _game.bitboard.undo_move()

                if depth == self.max_depth:
                    print("Ruch:", move, "Ocena ruchu:",
                          new_value, "| player X")
                if new_value > value:
                    value = new_value
                    best_move = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    flag = "LOWER"  # Beta-cutoff
                    break
            else:
                # If we never did a cutoff, the value is exact
                flag = "EXACT"

            # Store in TT
            self.transposition_table[current_hash] = (
                value, depth, flag, best_move)
            return value, best_move

        else:
            value = INT_MAX
            flag = "LOWER"  # Assume lower-bound until proven otherwise
            for move in moves:
                _game.bitboard.make_move(move)
                # Make sure to undo the move even if the function raises an error
                try:
                    new_value, _ = await self.min_max(
                        _game, depth - 1, alpha, beta, True, start_time, time_limit)
                finally:
                    _game.bitboard.undo_move()

                if depth == self.max_depth:
                    print("Ruch:", move, "Ocena ruchu:",
                          new_value, "| player O")
                if new_value < value:
                    value = new_value
                    best_move = move
                beta = min(beta, value)
                if beta <= alpha:
                    flag = "UPPER"  # Alpha-cutoff
                    break
            else:
                # If we never did a cutoff, the value is exact
                flag = "EXACT"

            # Store in TT
            self.transposition_table[current_hash] = (
                value, depth, flag, best_move)
            return value, best_move

    async def iterative_deepening(self, _game: Game) -> AsyncGenerator[int | None]:
        """Iterative Deepening: incrementally increase depth."""
        self.__explored_nodes = 0
        best_move_overall = None
        start_time = time.time()

        for depth in range(1, self.max_depth + 1):
            self.transposition_table.clear()
            try:
                value, best_move = await self.min_max(_game, depth, INT_MIN, INT_MAX,
                                                      not _game.bitboard.turn(), start_time,
                                                      self.time_limit)
            except asyncio.CancelledError:
                print("Timed out!")
                break
            if best_move is not None:
                best_move_overall = best_move

            print(f"Depth {depth}: Best Move = {best_move}, Score = {value}")

            yield best_move_overall

        print(f"Explored nodes: {self.__explored_nodes}")

    async def make_move_async(self, _game: Game) -> int | None:
        """Make a move in the time given."""
        best_move = None
        async for _move in self.iterative_deepening(_game):
            best_move = _move
        return best_move

    def make_move(self, _game: Game) -> int | None:
        """Makes the async move."""
        return asyncio.run(self.make_move_async(_game))


def main():
    """The main function."""
    game = Game()
    ai_o = Player(max_depth=20, time_limit=5)
    ai_x = Player(max_depth=20, time_limit=5)

    print("Rozpoczyna się gra AI vs AI")
    print(game.bitboard)
    while True:
        is_terminal, turn = game.bitboard.is_terminal()
        if is_terminal:
            break

        if game.bitboard.turn() == 0:
            start = time.time()
            move = ai_x.make_move(game)
            end = time.time()
            print(f"Czas wykonania: {end - start} sekund")
            print(f"X wybiera kolumnę {move}")
        else:
            start = time.time()
            move = ai_o.make_move(game)
            end = time.time()
            print(f"Czas wykonania: {end - start} sekund")
            print(f"O wybiera kolumnę {move}")

        assert move is not None
        game.bitboard.make_move(move)
        print(game.bitboard)
        print("------------------------------")
        print()

    if turn == 0:
        print("Wygrywa X!")
    else:
        print("Wygrywa O!")


if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumulative")
    # stats.print_stats(20)
