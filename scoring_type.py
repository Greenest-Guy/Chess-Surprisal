from stockfish import Stockfish
from dotenv import load_dotenv
import chess
import os

if __name__ == '__main__':
    load_dotenv()
    stockfish_path = os.getenv('STOCKFISH_PATH')
    stockfish = Stockfish(path=stockfish_path)
    stockfish.set_depth(15)

    # Black Winning & black to move
    position = "2k2r1r/ppp2p1p/6p1/6B1/8/2N4P/PP3PP1/6K1 b - - 0 2"

    board = chess.Board(position)

    stockfish.set_fen_position(board.fen())

    eval = stockfish.get_evaluation()['value']

    if eval < 0:  # white perspective
        print("Stockfish Absolute Scoring")

    elif eval > 0:
        print("Stockfish Relative Scoring")

    else:
        print("Error")
