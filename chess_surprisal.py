from stockfish import Stockfish
from dotenv import load_dotenv
import numpy as np
import chess


class surprisal:
    def __init__(self, stockfish: Stockfish, san_moves: list, delta0: int, info: bool):
        self.stockfish = stockfish
        self.san_moves = san_moves
        self.delta0 = delta0
        self.info = info

    def calculate_surprisal(self):
        board = chess.Board()
        self.stockfish.set_fen_position(board.fen())

        probabilities = []
        for i in range(len(self.san_moves)):
            white_to_move: bool = board.turn
            move = self.san_moves[i]

            # Gets ALL legal moves (218 is considered the maximum number of legal moves in a legal chess position.)
            legal_moves = self.stockfish.get_top_moves(256)

            # Gets the centipawn evaluation of all legal moves.
            evals = []
            for i in legal_moves:
                evals.append(self.get_centipawn(i))

            # Gets the centipawn value of Ei (played move) from all legal moves.
            uci_move = board.parse_san(move).uci()

            for i in legal_moves:
                if i['Move'] == uci_move:
                    Ei = self.get_centipawn(i)

            # If black to move, negate centipawn evals to switch to relative perspective.
            if not white_to_move:
                Ei = -Ei
                evals = [-i for i in evals]

            # Calculates softmax probability.
            move_probability = self.softmax_probability_all(
                evals, Ei, self.delta0)

            probabilities.append(move_probability)

            if self.info:
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
    stockfish = Stockfish(
        path=r"C:\Users\LegoB\Desktop\Cybersicherheit\VSCode\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe")

    stockfish.set_depth(15)

    moves = ['d4', 'Nf6', 'c4', 'e6', 'Nc3', 'Bb4', 'Nf3', 'c5', 'g3', 'cxd4', 'Nxd4', 'Ne4', 'Qc2', 'Nxc3', 'bxc3', 'Be7', 'Bg2', 'O-O', 'O-O', 'a6', 'Rd1', 'Qc7', 'Qb3', 'd6', 'Be3', 'Nd7', 'Rab1', 'Rb8', 'Qc2', 'Nc5', 'Nb3', 'b6', 'Nxc5', 'bxc5', 'Rxb8', 'Qxb8', 'Rb1', 'Qc7', 'Qa4', 'Bf6', 'Rb3',
                   'h6', 'Qc6', 'Qxc6', 'Bxc6', 'Bd8', 'Bf4', 'Bc7', 'Bb7', 'g5', 'Be3', 'Bd7', 'Bf3', 'Bc8', 'Bb7', 'Bd7', 'Bxa6', 'Ra8', 'Rb7', 'Rxa6', 'Rxc7', 'Ba4', 'h4', 'gxh4', 'gxh4', 'Kg7', 'Rb7', 'Bd1', 'Rb2', 'Ra4', 'Rd2', 'Rxc4', 'Rxd1', 'd5', 'Ra1', 'Rxc3', 'a4', 'd4', 'a5', 'dxe3', 'a6', 'exf2+', 'Kxf2']

    obj = surprisal(
        stockfish, moves, 10, True)

    print(obj.calculate_surprisal())
