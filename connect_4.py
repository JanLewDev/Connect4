"""Connect 4 AI using Minmax with alpha-beta pruning."""
from sys import maxsize
import time
import asyncio
from typing import AsyncGenerator
from bitboard import BitBoard

INT_MAX = maxsize
INT_MIN = -maxsize


class Game:
    """Game class for Connect 4."""
    n_rows: int
    n_columns: int
    winning_length: int
    board: list[list[int]]
    move_history: list[int]
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

        score = (p0_potential_lines - p1_potential_lines) +  (player1_threats - player0_threats)
        return score

    def set_bitboard(self) -> None:
        """Set the bitboard from the board."""
        for _move in self.move_history:
            self.bitboard.make_move(_move)


class Player:
    """Player class for Connect 4."""
    __explored_nodes: int = 0

    def __init__(self, max_depth: int = 5, time_limit: float | None = None) -> None:
        if time_limit is None:
            time_limit = 5.0
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
                    return INT_MAX - (self.max_depth - depth), None
                if _turn == 1:
                    return INT_MIN + (self.max_depth - depth), None
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

    async def iterative_deepening(self, _game: Game):
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

        assert best_move is not None
        _game.move_history.append(best_move)
        return best_move

    def make_move(self, _game: Game) -> int | None:
        """Makes the async move."""
        _game.bitboard = BitBoard(
            height=_game.n_rows, width=_game.n_columns, winning_length=_game.winning_length)
        _game.set_bitboard()
        return asyncio.run(self.make_move_async(_game))
