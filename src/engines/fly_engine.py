import torch
import torch.nn.functional as F
import chess

from src.fly.single.action_chooser import ActionChooser
from src.fly.single.tokenizer import tokenize
from src.engines import engine


class FlyEngine:
    """Adapter class that makes ActionChooser compatible with chess.engines.Engine interface."""

    def __init__(self, path, device=None):
        """Initialize the FlyChessEngine.

        Args:
            path: Path to the model checkpoint
            device: Device to run the model on (cuda, mps, or cpu)
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        self.action_chooser = ActionChooser(path, device)
        self.device = device

    def analyse(self, board: chess.Board) -> engine.AnalysisResult:
        """Analyze the board position and return evaluation results.

        Args:
            board: Chess board position

        Returns:
            Dictionary containing analysis results
        """
        # Store the original board
        original_board = board.copy()

        # Get probabilities for each legal move
        all_move_data = []
        legal_moves = list(board.legal_moves)
        move_probs = {}

        # Process each legal move
        for move in legal_moves:
            board.push(move)
            data = tokenize(board.fen())
            data = torch.from_numpy(data).float()
            data = data.to(self.device)
            all_move_data.append(data)
            board.pop()

        # Get model outputs for all moves
        if all_move_data:
            data = torch.stack(all_move_data, dim=0)
            output = self.action_chooser.model(data)
            probs = F.softmax(output, dim=0)

            # Store probabilities for each move
            for i, move in enumerate(legal_moves):
                move_probs[move.uci()] = probs[i].tolist()

        # Return analysis results
        return {"fen": original_board.fen(), "move_probs": move_probs}

    def play(self, board: chess.Board) -> chess.Move:
        """Return the best move for the given board position.

        Args:
            board: Chess board position

        Returns:
            Best move according to the model
        """
        return self.action_chooser.play(board)
