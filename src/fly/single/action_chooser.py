import torch
import torch.nn.functional as F
from typing import Any, Mapping

from src.fly.single.tokenizer import tokenize

# Type alias to match the chess engine interface
AnalysisResult = Mapping[str, Any]


class ActionChooser:
    def __init__(self, path, device):
        checkpoint = torch.load(path, weights_only=False)
        self.device = device

        # Use the original model directly; now with channel adaptation
        self.model = checkpoint["model"].to(self.device)

        # Print information about the model
        print(f"Loaded model with input shape adaptation")

        # Check if we need to add a channel adapter for older models
        # Note: This is handled inside the model classes now

    def play(self, board):
        all_move_data = []
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            board.push(move)
            data = tokenize(board.fen())
            data = torch.from_numpy(data).float()
            data = data.to(self.device)
            all_move_data.append(data)
            board.pop()
        data = torch.stack(all_move_data, dim=0)
        output = self.model(data)
        probs = F.softmax(output, dim=1)
        choices = torch.argmin(
            probs, dim=0
        ).tolist()  # This is changed to min because ()
        return legal_moves[choices[0]]
