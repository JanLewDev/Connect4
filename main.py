"""Connect 4 AI using Minmax with alpha-beta pruning."""
import sys
import cProfile
import pstats

INT_MAX = sys.maxsize
INT_MIN = -sys.maxsize


class BitBoard:
    """Board class for Connect 4."""
    height: int
    width: int
    winning_length: int
    move_history: list[int]
    bitboard: list[int]
    __counter: int
    __heights: list[int]  # current height

    def __init__(self, height: int = 6, width: int = 7, winning_length: int = 4) -> None:
        self.height = height
        self.width = width
        self.winning_length = winning_length
        self.move_history = []
        self.__counter = 0
        self.bitboard = [0] * 2  # We always have two players
        # positions of the next possible move in each column
        self.__heights = [i * (height + 1) for i in range(width)]
        assert len(self.__heights) == width

    def make_move(self, col: int) -> None:
        """Make a move in the column."""
        # assert 0 <= col < self.width
        _move: int = 1 << self.__heights[col]
        self.bitboard[self.turn()] ^= _move
        self.__heights[col] += 1
        self.move_history.append(col)
        self.__counter += 1

    def undo_move(self) -> None:
        """Undo the last move."""
        col = self.move_history.pop()
        self.__counter -= 1
        self.__heights[col] -= 1
        _move: int = 1 << self.__heights[col]
        self.bitboard[self.turn()] ^= _move

    def is_winning(self, bitboard: int | None = None) -> bool:
        """Check if the passed board is winning or the current player has won."""
        if bitboard is None:
            bitboard = self.bitboard[self.turn() ^ 1]
        # corresponding to vertical, diagonal \, horizontal, diagonal /
        directions = [1, self.height, self.height + 1, self.height + 2]
        for shift_length in directions:
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
            self.width, (self.__counter & 1)

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
        """Evaluate the board using bitboards."""
        player0_bitboard = self.bitboard.bitboard[0]
        player1_bitboard = self.bitboard.bitboard[1]

        H = self.bitboard.height
        W = self.bitboard.width
        K = self.bitboard.winning_length

        p0_potential_lines = 0
        p1_potential_lines = 0
        player0_threats = 0
        player1_threats = 0

        # Directions for shifts
        # Vertical: 1
        # Horizontal: H + 1
        # Diagonal / (up-right): H + 2
        # Diagonal \\ (down-right, effectively): H
        shifts = [1, H + 1, H + 2, H]

        for r_start in range(H):
            for c_start in range(W):
                for shift in shifts:
                    line_mask = 0

                    current_positions = []

                    possible_line = True
                    bit_r, bit_c = -1, -1
                    for i in range(K):
                        if shift == 1:  # Vertical
                            bit_r, bit_c = r_start + i, c_start
                        elif shift == H + 1:  # Horizontal
                            bit_r, bit_c = r_start, c_start + i
                        elif shift == H + 2:  # Diagonal / (up-right)
                            bit_r, bit_c = r_start + i, c_start + i
                        # Diagonal \\ (down-right relative to start)
                        elif shift == H:
                            bit_r, bit_c = r_start - i, c_start + i

                        if not (0 <= bit_r < H and 0 <= bit_c < W):
                            possible_line = False
                            break

                        bit_pos = bit_c * (H + 1) + bit_r
                        line_mask |= (1 << bit_pos)
                        current_positions.append((bit_r, bit_c))

                    if not possible_line or len(current_positions) != K:
                        continue

                    x_pieces = self._count_set_bits(
                        line_mask & player0_bitboard)
                    o_pieces = self._count_set_bits(
                        line_mask & player1_bitboard)
                    empty_spots = K - x_pieces - o_pieces

                    if x_pieces > 0 and o_pieces > 0:
                        continue

                    if o_pieces == 0:
                        p0_potential_lines += 1
                        if x_pieces == K - 1 and empty_spots == 1:
                            player0_threats += 3
                        elif x_pieces == K - 2 and empty_spots == 2:
                            player0_threats += 2
                        elif K >= 3 and x_pieces == K - 3 and empty_spots == 3:
                            player0_threats += 1

                    if x_pieces == 0:
                        p1_potential_lines += 1
                        if o_pieces == K - 1 and empty_spots == 1:
                            player1_threats += 3
                        elif o_pieces == K - 2 and empty_spots == 2:
                            player1_threats += 2
                        elif K >= 3 and o_pieces == K - 3 and empty_spots == 3:
                            player1_threats += 1

        # Score = (p0_potential_lines - p1_potential_lines) + (player1_threats - player0_threats)
        score = (p0_potential_lines - p1_potential_lines) + \
            (player1_threats - player0_threats)
        return score


class Player:
    """Player class for Connect 4."""
    __explored_nodes: int = 0

    def __init__(self, max_depth: int = 7) -> None:
        self.max_depth = max_depth

    @staticmethod
    def order_moves(moves: list[int], cols: int) -> list[int]:
        """Simple rule of choosing the moves closer to the centre first."""
        return sorted(moves, key=lambda x: x if x < cols // 2 else cols - 1 - x, reverse=True)

    def min_max(self, _game: Game, depth: int, alpha: int, beta: int, maximizing: bool) \
            -> tuple[int, int | None]:
        """Minimax algorithm with alpha-beta pruning."""
        self.__explored_nodes += 1
        _is_terminal, _turn = _game.bitboard.is_terminal()
        if depth == 0 or _is_terminal:
            if _is_terminal:
                if _turn == 0:
                    return INT_MAX-1, None
                if _turn == 1:
                    return INT_MIN+1, None
                return 0, None
            return _game.eval(), None
        moves = self.order_moves(_game.bitboard.list_moves(), _game.n_columns)
        best_move = None
        if maximizing:
            value = INT_MIN
            for _move in moves:
                _game.bitboard.make_move(_move)
                new_value, _ = self.min_max(_game, depth-1, alpha, beta, False)
                _game.bitboard.undo_move()
                if depth == self.max_depth:
                    print("Ruch:", _move, "Ocena ruchu:",
                          new_value, "| player X")
                if new_value > value:
                    value, best_move = new_value, _move
                alpha = max(alpha, value)
            return value, best_move

        value = INT_MAX
        for _move in moves:
            _game.bitboard.make_move(_move)
            new_value, _ = self.min_max(_game, depth-1, alpha, beta, True)
            _game.bitboard.undo_move()
            if depth == self.max_depth:
                print("Ruch:", _move, "Ocena ruchu:",
                      new_value, "| player O")
            if new_value < value:
                value, best_move = new_value, _move
            beta = min(beta, value)
            if beta <= alpha:
                break
        return value, best_move

    def make_move(self, _game: Game) -> int | None:
        """Call the min_max function and return the best move."""
        self.__explored_nodes = 0
        _, _move = self.min_max(_game, self.max_depth, INT_MIN, INT_MAX,
                                not _game.bitboard.turn())
        print(f"Przeszukanych węzłów: {self.__explored_nodes}")
        return _move


def print_board(game: Game):
    symbols = {-1: '.', 0: 'X', 1: 'O'}
    print(' '.join(map(str, range(game.n_columns))))
    for r in range(game.n_rows):
        print(' '.join(symbols[game.board[r][c]]
              for c in range(game.n_columns)))


def main():
    game = Game()
    ai_o = Player(max_depth=7)
    ai_x = Player(max_depth=7)
    game.board = [
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1]
    ]

    print("Rozpoczyna się gra AI vs AI")
    print(game.bitboard)
    while True:
        is_terminal, turn = game.bitboard.is_terminal()
        if is_terminal:
            break

        if game.bitboard.turn() == 0:
            move = ai_x.make_move(game)
            print(f"X wybiera kolumnę {move}")
        else:
            move = ai_o.make_move(game)
            print(f"O wybiera kolumnę {move}")

        assert move is not None
        game.bitboard.make_move(move)
        print(game.bitboard)
        print("------------------------------")
        print()

    turn ^= 1
    if turn == 0:
        print("Wygrywa X!")
    else:
        print("Wygrywa O!")
if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)