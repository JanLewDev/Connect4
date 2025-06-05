import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from bitboard import BitBoard
from typing import List, Tuple

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

class Connect4NN(nn.Module):

    def __init__(self, height: int = 6, width: int = 7):
        super().__init__()
        self.height = height
        self.width = width

        self.conv1 = nn.Conv2d(
            in_channels=2, out_channels=128, kernel_size=4, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(64)

        flatten_size = 64 * (self.height-1) * (self.width-1)

        self.fc1 = nn.Linear(flatten_size, 256)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(256, 1)

        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        x: tensor o kształcie (batch_size, 2, H, W)
        zwraca tensor o kształcie (batch_size,), z wartościami w [-1,1].
        """

        x = self.conv1(x) 
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.conv2(x)         
        x = self.bn2(x)
        x = nn.functional.relu(x)

        x = x.view(x.size(0), -1)  

        x = self.fc1(x)             
        x = nn.functional.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)             
        x = self.tanh(x)            
        return x.squeeze(1)

if __name__ == "__main__":
    NUM_SAMPLES = 10000
    PLAYOUTS = 100
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 1e-4

    TRAIN_SPLIT = 0.70
    VALID_SPLIT = 0.15
    TEST_SPLIT = 0.15

    dataset = Connect4Dataset(
        num_samples=NUM_SAMPLES,
        playouts_per_position=PLAYOUTS,
        height=6,
        width=7,
        winning_length=4,
    )

    total_size = len(dataset)
    n_test = int(total_size * TEST_SPLIT)
    n_valid = int(total_size * VALID_SPLIT)
    n_train = total_size - n_valid - n_test

    train_set, valid_set, test_set = random_split(
        dataset, [n_train, n_valid, n_test]
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Connect4NN(height=6, width=7).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for boards, targets in train_loader:
            boards = boards.to(device)      
            targets = targets.to(device)    

            optimizer.zero_grad()
            outputs = model(boards)         
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * boards.size(0)

        epoch_train_loss = running_loss / n_train

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for boards, targets in valid_loader:
                boards = boards.to(device)
                targets = targets.to(device)
                outputs = model(boards)
                loss = criterion(outputs, targets)
                valid_loss += loss.item() * boards.size(0)

        epoch_valid_loss = valid_loss / n_valid

        print(
            f"Epoch [{epoch}/{EPOCHS}]  "
            f"Train Loss: {epoch_train_loss:.4f}  "
            f"Valid Loss: {epoch_valid_loss:.4f}"
        )

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for boards, targets in test_loader:
            boards = boards.to(device)
            targets = targets.to(device)
            outputs = model(boards)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * boards.size(0)

    test_loss /= n_test
    print(f"\nTest Loss: {test_loss:.4f}")