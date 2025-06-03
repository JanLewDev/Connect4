import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

class BitBoard:
    """Board class for Connect 4."""
    height: int
    width: int
    winning_length: int
    move_history: list[int]
    bitboard: list[int]
    counter: int
    heights: list[int]  # current height
    zobrist_table: list[list[list[int]]]

    def __init__(self, height: int = 6, width: int = 7, winning_length: int = 4) -> None:
        self.height = height
        self.width = width
        self.winning_length = winning_length
        self.move_history = []
        self.counter = 0
        self.bitboard = [0] * 2  # We always have two players
        # positions of the next possible move in each column
        self.heights = [i * (height + 1) for i in range(width)]
        self.shifts = [1, self.height, self.height + 1, self.height + 2]
        # Initialize Zobrist table: [player(0/1)][col][row]
        self.zobrist_table = [[[random.getrandbits(64) for _ in range(
            height)] for _ in range(width)] for _ in range(2)]
        self.current_hash = 0
        assert len(self.heights) == width

    def make_move(self, col: int) -> None:
        """Make a move in the column."""
        # assert 0 <= col < self.width
        row = self.heights[col] - col * (self.height + 1)
        player = self.turn()
        # Update bitboard
        _move: int = 1 << self.heights[col]
        self.bitboard[player] ^= _move
        # Update Zobrist hash
        self.current_hash ^= self.zobrist_table[player][col][row]
        self.heights[col] += 1
        self.move_history.append(col)
        self.counter += 1

    def undo_move(self) -> None:
        """Undo the last move."""
        col = self.move_history.pop()
        self.counter -= 1
        self.heights[col] -= 1
        # Determine row index after undo (the position we remove)
        row = self.heights[col] - col * (self.height + 1)
        player = self.turn()
        _move: int = 1 << self.heights[col]
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
            if not self.__top_mask(col) & (1 << self.heights[col]):
                moves.append(col)
        return moves

    def is_terminal(self) -> tuple[bool, int]:
        """Check if the board is terminal and return the last move's player."""
        return self.is_winning() or \
            self.counter == self.height * \
            self.width, (self.turn() ^ 1)  # self.turn()?

    def turn(self) -> int:
        """Return the current player."""
        return self.counter & 1

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

class Connect4Dataset(Dataset):
  """Dataset generated using random simulations"""
  def __init__(self, num_samples: int, playouts_per_position: int, height: int = 7, width: int = 7, winning_length: int = 4):
    self.num_samples = num_samples
    self.positions = []
    self.labels = []
    self.playouts= playouts_per_position
    self.height = height
    self.width = width
    self.winning_length = winning_length
    self.generate_data()

  def generate_data(self):
    for _ in range(self.num_samples):
      game = BitBoard(self.height, self.width, self.winning_length)

      # Generate random board
      n_moves = random.randint(0, self.height * self.width)
      for _ in range(n_moves):
        is_terminal, _ = game.is_terminal()
        if is_terminal:
          break
        legal_moves = game.list_moves()
        if not legal_moves:
          break
        move_to_apply = random.choice(legal_moves)
        game.make_move(move_to_apply)

      # Get features
      features = self.board_to_tensor(game.bitboard)
      self.positions.append(features)

      # Get label
      score = 0.0
      for _ in range(self.playouts):
        result = self.simulate_random_game(game)
        score += result
      label = score / self.playouts
      self.labels.append(label)

  def simulate_random_game(self, game: BitBoard) -> float:
    game_clone = BitBoard(self.height, self.width, self.winning_length)
    game_clone.bitboard[0] = game.bitboard[0]
    game_clone.bitboard[1] = game.bitboard[1]
    game_clone.move_history = game.move_history
    game_clone.heights = game.heights
    game_clone.counter = game.counter

    while True:
      is_terminal, winner = game_clone.is_terminal()
      if is_terminal:
        if winner == 0:
          return 1.0
        elif winner == 1:
          return -1.0
        else:
          return 0.0
      legal_moves = game_clone.list_moves()
      if not legal_moves:
        return 0.0
      move = random.choice(legal_moves)
      game_clone.make_move(move)
      

  def board_to_tensor(self, bitboards: List[int]) -> torch.Tensor:
    """Converts two bitboards into two tensors with dimensions 2 x H x W"""
    H = self.height
    W = self.width
    board_tensor = torch.zeros((2, H, W), dtype=torch.float32)
    for player in (0, 1):
      bb = bitboards[player]
      for col in range(H):
        for row in range(W):
          bit_pos = col * (H + 1) + row
          if (1 << bit_pos) & bb:
            board_tensor[player, row, col] = 1.0
    return board_tensor
  def __len__(self) -> int:
    return self.num_samples
  def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return self.positions[idx], torch.tensor(self.labels[idx])
