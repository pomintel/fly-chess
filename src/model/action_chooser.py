import torch
from src.chess_utils.tokenizer import tokenize
from src.basics import get_device
import torch.nn.functional as F

class ActionChooser():
    def __init__(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.device = get_device()
        self.model = checkpoint["model"].to(self.device)
        self.config = checkpoint["config"]
        self.model_name = self.model.__class__.__name__

    def play(self, board):
        all_move_data = []
        legal_moves = list(board.legal_moves)
        for move in legal_moves :
            board.push(move)
            data = tokenize(board.fen(), self.config)
            data = torch.from_numpy(data).float()
            data = data.to(self.device)
            all_move_data.append(data)
            board.pop()
        data = torch.stack(all_move_data, dim=0)
        output = self.model(data)
        probs = F.softmax(output, dim=1)
        choices = torch.argmin(probs, dim=0).tolist()
        #This is changed to min because It's the opponent's winning prob
        return legal_moves[choices[0]]
