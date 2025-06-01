# Connect 4 AI in Python

## Description

Minimax algorithm with alpha-beta pruning.

## Usage

```bash
pip3 install -r requirements.txt
python3 main.py
```

## More detailed description

The game state is stored using a Bitboard consisting of Width * (Height + 1) bits. One extra bit at the top to encoode full columns. For a standard 7 x 6 board, this would be 49 bits like so:

```console
.  .  .  .  .  .  .
5 12 19 26 33 40 47
4 11 18 25 32 39 46
3 10 17 24 31 38 45
2  9 16 23 30 37 44
1  8 15 22 29 36 43
0  7 14 21 28 35 42 
```

We will be storing two bitboards, one for each player.

The class structure needed to conform to standards set by the Professor.
