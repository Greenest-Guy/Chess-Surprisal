from stockfish import Stockfish
from dotenv import load_dotenv
import numpy as np
import chess
import os


class chess_surprisal:
    def __init__(self, stockfish: Stockfish, san_moves: list, delta0: int):
        self.stockfish = stockfish
        self.san_moves = san_moves
        self.delta0 = delta0

    def calculate_surprisal(self, info=False, absolute_scoring=False):
        board = chess.Board()
        self.stockfish.set_fen_position(board.fen())

        probabilities = []

        # iterate through each chess move
        for i in range(len(self.san_moves)):
            white_to_move: bool = board.turn
            move = self.san_moves[i]

            # Gets ALL legal moves (218 is considered the maximum number of legal moves in a legal chess position.)
            legal_moves = self.stockfish.get_top_moves(256)

            # Gets the centipawn evaluation of all legal moves.
            evals = []
            for j in legal_moves:
                evals.append(self.get_centipawn(j))

            # Gets the centipawn value of Ei (played move) from all legal moves.
            uci_move = board.parse_san(move).uci()

            for j in legal_moves:
                if j['Move'] == uci_move:
                    Ei = self.get_centipawn(j)
                    break

            # If black to move, negate centipawn evals to switch to relative perspective. ONLY IF STOCKFISH USES ABSOLUTE SCORING
            if absolute_scoring and not white_to_move:
                Ei = -Ei
                evals = [-j for j in evals]

            # Calculates softmax probability.
            move_probability = self.softmax_probability_all(
                evals, Ei, self.delta0)

            probabilities.append(move_probability)

            if info:
                print(
                    f"Best Move: {legal_moves[0]['Move']} at eval {evals[0]}")
                print(f"Played Move: {move} at eval {Ei}")
                print(f"Move Probability: {move_probability * 100}%\n")

            # Try to make next move
            try:
                board.push_san(move)
                self.stockfish.set_fen_position(board.fen())

            except IndexError:  # If there are no more moves break.
                break

        return self.calculate_total_information_cost(probabilities)

    @staticmethod
    def get_centipawn(move_data: dict):
        if move_data['Mate'] is not None:
            # Mate value = 20000
            if move_data['Mate'] > 0:
                return 20000
            else:
                return -20000

        return move_data['Centipawn']

    @staticmethod
    # Creates probability considering all legal moves.
    def softmax_probability_all(evals: list, Ei: int, delta0: int):
        evals = np.array(evals, dtype=float)

        # Temperature Scaling
        scaled_evals = evals / delta0
        scaled_Ei = Ei / delta0

        # prevents exponential overflow
        max_val = np.max(scaled_evals)

        numerator = np.exp(scaled_Ei - max_val)
        denominator = np.sum(np.exp(scaled_evals - max_val))
        return numerator / denominator

    @staticmethod
    def calculate_total_information_cost(probabilities: list):
        probabilities_np = np.array(probabilities, dtype=float)

        # Prevent log divide by zero
        probabilities_np = np.clip(probabilities_np, 1e-15, 1.0)

        total_surprisal = -np.sum(np.log2(probabilities_np))
        return total_surprisal


if __name__ == '__main__':
    load_dotenv()

    stockfish_path = os.getenv('STOCKFISH_PATH')

    stockfish = Stockfish(path=stockfish_path)

    stockfish.set_depth(12)

    surprisal_calculator = chess_surprisal(stockfish, ['e4', 'e5'], 100)

    print(surprisal_calculator.calculate_surprisal(True))
