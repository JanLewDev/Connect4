"""Main file utilising the connect_4.py module."""
import time
import argparse
from connect_4 import Game, Player

# For now only supports ai_vs_ai mode
MODES = ["ai_vs_ai"]

parser = argparse.ArgumentParser(description="Connect 4 AI")
parser.add_argument("-r", "--rows", type=int, default=6, help="Number of rows")
parser.add_argument("-c", "--columns", type=int, default=7,
                    help="Number of columns")
parser.add_argument("-w", "--winning_length", type=int,
                    default=4, help="Winning length")
parser.add_argument("-m", "--mode", type=str, choices=MODES, default="ai_vs_ai",
                    help="Mode of the game")
parser.add_argument("-d", "--max_depth", nargs="*", type=int, default=[20, 20],
                    help="Max depth for each player")
parser.add_argument("-t", "--time_limit", nargs="*", type=float, default=[None, None],
                    help="Time limit for each player")
args = parser.parse_args()

max_depths = [20, 20]
if args.max_depth and len(args.max_depth) >= 1:
    max_depths[0] = args.max_depth[0]
    if len(args.max_depth) >= 2:
        max_depths[1] = args.max_depth[1]

time_limits = [None, None]
if args.time_limit and len(args.time_limit) >= 1:
    time_limits[0] = args.time_limit[0]
    if len(args.time_limit) >= 2:
        time_limits[1] = args.time_limit[1]


def main():
    """The main function."""
    game = Game(args.rows, args.columns, args.winning_length)
    ai_x = Player(max_depth=max_depths[0], time_limit=time_limits[0])
    ai_o = Player(max_depth=max_depths[1], time_limit=time_limits[1])

    print("AI vs AI game starts")
    print("Starting with arguments:")
    print(f"Rows: {args.rows}")
    print(f"Columns: {args.columns}")
    print(f"Winning length: {args.winning_length}")
    print(f"Mode: {args.mode}")
    print(f"Max depths: {max_depths}")
    print(f"Time limits: {time_limits}")
    print(game.bitboard)
    while True:
        is_terminal, turn = game.bitboard.is_terminal()
        if is_terminal:
            break

        if game.bitboard.turn() == 0:
            start = time.time()
            move = ai_x.make_move(game)
            end = time.time()
            print(f"Time taken: {end - start} seconds")
            print(f"X chooses column {move}")
        else:
            start = time.time()
            move = ai_o.make_move(game)
            end = time.time()
            print(f"Time taken: {end - start} seconds")
            print(f"O chooses column {move}")

        assert move is not None
        game.bitboard.make_move(move)
        print(game.bitboard)
        print("------------------------------")
        print()

    if turn == 0:
        print("X wins!")
    else:
        print("O wins!")


if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats("cumulative")
    # stats.print_stats(20)
