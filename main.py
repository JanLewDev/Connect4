import sys
import time

INT_MAX = sys.maxsize
INT_MIN = -sys.maxsize


class Game:
    n_rows: int
    n_columns: int
    winning_length: int
    board: list[list[int]]
    move_history: list[tuple[int, int]]
    current_player: int

    def __init__(self, n_rows=7, n_columns=7, winning_length=4, current_player=0) -> None:
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.winning_length = winning_length
        self.current_player = current_player
        self.board = [[-1 for _ in range(n_columns)] for _ in range(n_rows)]
        self.move_history = []

    def legal_moves(self) -> list[int]:
        return [c for c in range(self.n_columns) if self.board[0][c] == -1]

    def apply_move(self, col: int | None) -> None:
        if col is None:
            return
        for r in range(self.n_rows-1, -1, -1):
            if self.board[r][col] == -1:
                self.board[r][col] = self.current_player
                self.move_history.append((r, col))
                self.current_player = 1 - self.current_player
                return

    def undo_move(self) -> None:
        r, c = self.move_history.pop()
        self.board[r][c] = -1
        self.current_player = 1 - self.current_player

    def check_win(self, player):
        for r in range(self.n_rows):
            for c in range(self.n_columns):
                if self.board[r][c] != player:
                    continue
                # four directions
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    count = 1
                    for i in range(1, self.winning_length):
                        nr, nc = r + dr * i, c + dc * i
                        if 0 <= nr < self.n_rows and 0 <= nc < self.n_columns and self.board[nr][nc] == player:
                            count += 1
                        else:
                            break
                    if count >= self.winning_length:
                        return True
        return False

    def is_terminal(self) -> bool:
        return self.check_win(0) or self.check_win(1) or len(self.legal_moves()) == 0

    def winner(self) -> int:
        if self.check_win(0):
            return 0
        if self.check_win(1):
            return 1
        return -1

    def eval(self) -> int:
        p0_count = p1_count = 0
        threat0 = threat1 = 0
        for r in range(self.n_rows):
            for c in range(self.n_columns):
                for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                    cells = []
                    for i in range(self.winning_length):
                        rr, cc = r + dr*i, c + dc*i
                        if 0 <= rr < self.n_rows and 0 <= cc < self.n_columns:
                            cells.append(self.board[rr][cc])
                        else:
                            break
                    if len(cells) != self.winning_length:
                        continue
                    if 0 in cells and 1 in cells:
                        continue
                    empty = cells.count(-1)
                    x_count = cells.count(0)
                    o_count = cells.count(1)
                    if x_count == 0:
                        p1_count += 1
                    if o_count == 0:
                        p0_count += 1
                    if x_count == self.winning_length - 1 and empty == 1:
                        threat1 += 3
                    if o_count == self.winning_length - 1 and empty == 1:
                        threat0 += 3
                    if x_count == self.winning_length - 2 and empty == 2:
                        threat1 += 2
                    if o_count == self.winning_length - 2 and empty == 2:
                        threat0 += 2
                    if x_count == self.winning_length - 3 and empty == 3:
                        threat1 += 1
                    if o_count == self.winning_length - 3 and empty == 3:
                        threat0 += 1
        return 1*(p0_count - p1_count) + (threat0 - threat1) * 1


class Player:
    def __init__(self, max_depth: int = 4) -> None:
        self.max_depth = max_depth

    def min_max(self, game: Game, depth: int, alpha: int, beta: int, maximizing: bool) -> tuple[int, int | None]:
        if depth == 0 or game.is_terminal():
            if game.is_terminal():
                if game.winner() == 0:
                    return INT_MAX-1, None
                elif game.winner() == 1:
                    return INT_MIN+1, None
                else:
                    return 0, None
            return game.eval(), None
        moves = game.legal_moves()
        best_move = None
        if maximizing:
            value = INT_MIN
            for move in moves:
                game.apply_move(move)
                new_value, _ = self.min_max(game, depth-1, alpha, beta, False)
                game.undo_move()
                if depth == self.max_depth:
                    print("Ruch:", move, "Ocena ruchu:",
                          new_value, "| player X")
                if new_value > value:
                    value, best_move = new_value, move
                alpha = max(alpha, value)
            return value, best_move
        else:
            value = INT_MAX
            for move in moves:
                game.apply_move(move)
                new_value, _ = self.min_max(game, depth-1, alpha, beta, True)
                game.undo_move()
                if depth == self.max_depth:
                    print("Ruch:", move, "Ocena ruchu:",
                          new_value, "| player O")
                if new_value < value:
                    value, best_move = new_value, move
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value, best_move

    def make_move(self, game: Game) -> int | None:
        return self.min_max(game, self.max_depth, INT_MIN, INT_MAX, not game.current_player)[1] or None


def print_board(game: Game):
    symbols = {-1: '.', 0: 'X', 1: 'O'}
    print(' '.join(map(str, range(game.n_columns))))
    for r in range(game.n_rows):
        print(' '.join(symbols[game.board[r][c]]
              for c in range(game.n_columns)))


if __name__ == '__main__':
    game = Game()
    ai_o = Player(max_depth=11)
    ai_x = Player(max_depth=11)
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
    print_board(game)
    while not game.is_terminal():
        if game.current_player == 0:
            move = ai_x.make_move(game)
            print(f"X wybiera kolumnę {move}")
        else:
            move = ai_o.make_move(game)
            print(f"O wybiera kolumnę {move}")
        game.apply_move(move)
        print_board(game)
        time.sleep(1)
        print("------------------------------")
        print()
    winner = game.winner()
    if winner == -1:
        print("Remis!")
    else:
        print(f"Wygrywa {'X' if winner == 0 else 'O'}!")
