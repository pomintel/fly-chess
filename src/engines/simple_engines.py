# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Implements simple baseline agents for chess gameplay."""

import random
from collections.abc import Mapping
from typing import Any

import chess

AnalysisResult = Mapping[str, Any]


class RandomAgent:
    """A chess engine that makes completely random legal moves."""

    def analyse(self, board: chess.Board) -> AnalysisResult:
        """Returns a random analysis result."""
        return {"score": chess.engine.Score(cp=0)}

    def play(self, board: chess.Board) -> chess.Move:
        """Returns a random legal move from the given board."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")
        return random.choice(legal_moves)


class MaterialGreedyAgent:
    """A chess engine that greedily maximizes material gain on each move."""

    # Piece values according to standard chess evaluations
    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000,  # High value to prioritize king captures (checkmate)
    }

    def analyse(self, board: chess.Board) -> AnalysisResult:
        """Returns a simple material balance analysis."""
        material_balance = self._evaluate_material_balance(board)
        return {"score": chess.engine.Score(cp=material_balance)}

    def play(self, board: chess.Board) -> chess.Move:
        """Returns the move that maximizes immediate material gain."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available")

        best_move = None
        best_score = float("-inf")

        # Evaluate each move by the material gain it produces
        for move in legal_moves:
            # Try the move on a copy of the board
            board_copy = board.copy()
            board_copy.push(move)

            # Evaluate the resulting position
            # For captures, we look at the piece value difference
            capture_value = self._evaluate_move_value(board, move)

            if capture_value > best_score:
                best_score = capture_value
                best_move = move

        # If no captures available, pick a random move
        if best_move is None:
            return random.choice(legal_moves)

        return best_move

    def _evaluate_move_value(self, board: chess.Board, move: chess.Move) -> int:
        """Evaluate the material value of a move."""
        # Base value is 0 (non-capturing moves)
        value = 0

        # Capturing moves gain the value of the captured piece
        if board.is_capture(move):
            # If it's a capture, add the value of the captured piece
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                value += self.PIECE_VALUES[captured_piece.piece_type]

            # En passant capture
            if board.is_en_passant(move):
                value += self.PIECE_VALUES[chess.PAWN]

        # If the move is a promotion, add the difference in value
        # between the promoted piece and a pawn
        if move.promotion:
            value += self.PIECE_VALUES[move.promotion] - self.PIECE_VALUES[chess.PAWN]

        return value

    def _evaluate_material_balance(self, board: chess.Board) -> int:
        """Calculate the material balance for the current side to move."""
        balance = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.PIECE_VALUES[piece.piece_type]
                # Add value if it's our piece, subtract if opponent's
                balance += value if piece.color == board.turn else -value
        return balance
