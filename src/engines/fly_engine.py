import subprocess
import threading
import torch
import torch.nn.functional as F
import chess
import time
import os

from src.fly.single.action_chooser import ActionChooser
from src.fly.single.tokenizer import tokenize
from src.engines import engine


class FlyEngine(engine.Engine):
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


class UciFlyEngine:
    """UCI wrapper for FlyEngine that follows the UCI protocol."""

    def __init__(self, model_path=None):
        """Initialize the UCI FlyEngine.

        Args:
            model_path: Path to the model checkpoint
        """
        # Default model path if not specified
        if model_path is None:
            model_path = os.path.join(
                os.getcwd(),
                "../results/DPU_CNN_Unlearnable_1filters_2560000_trial1_2Timesteps-signed/model.pth",
            )

        # Create the engine
        self.engine = FlyEngine(model_path)
        self.board = chess.Board()
        self.debug = False
        self.name = "Floyd"
        self.author = "Pomintel"

    def process_command(self, command):
        """Process a UCI command and return the response.

        Args:
            command: UCI command string

        Returns:
            Response string according to UCI protocol
        """
        tokens = command.strip().split()
        if not tokens:
            return ""

        cmd = tokens[0]

        # UCI command
        if cmd == "uci":
            return (
                f"id name {self.name}\n"
                f"id author {self.author}\n"
                "option name Debug type check default false\n"
                "option name Model Path type string default results/DPU_CNN_Unlearnable_1filters_2560000_trial1_2Timesteps-signed/model.pth\n"
                "uciok"
            )

        # Ready command
        elif cmd == "isready":
            return "readyok"

        # Set options
        elif cmd == "setoption":
            if len(tokens) >= 5 and tokens[1] == "name" and tokens[3] == "value":
                option_name = tokens[2]
                option_value = tokens[4]

                if option_name == "Debug":
                    self.debug = option_value.lower() == "true"
                elif option_name == "Model Path":
                    self.engine = FlyEngine(option_value)

            return ""

        # New game
        elif cmd == "ucinewgame":
            self.board = chess.Board()
            return ""

        # Position setup
        elif cmd == "position":
            if len(tokens) < 2:
                return ""

            # Starting position
            if tokens[1] == "startpos":
                self.board = chess.Board()
                move_idx = 3 if len(tokens) > 2 and tokens[2] == "moves" else 0
            # Position from FEN
            elif tokens[1] == "fen":
                fen_parts = []
                i = 2
                while i < len(tokens) and tokens[i] != "moves":
                    fen_parts.append(tokens[i])
                    i += 1
                fen = " ".join(fen_parts)
                try:
                    self.board = chess.Board(fen)
                    move_idx = i + 1 if i < len(tokens) and tokens[i] == "moves" else 0
                except ValueError:
                    return f"info string Invalid FEN: {fen}"
            else:
                return ""

            # Apply moves if there are any
            if move_idx > 0 and move_idx < len(tokens):
                for move_str in tokens[move_idx:]:
                    try:
                        self.board.push_uci(move_str)
                    except ValueError:
                        return f"info string Invalid move: {move_str}"

            return ""

        # Generate a move
        elif cmd == "go":
            # Parse optional parameters (time control, etc.)
            # For simplicity, we're ignoring time control for now

            try:
                best_move = self.engine.play(self.board)
                if self.debug:
                    analysis = self.engine.analyse(self.board)
                    probs = analysis.get("move_probs", {})

                    # Output debug info about move probabilities
                    info_str = "info string Move probabilities:"
                    for move_uci, prob in probs.items():
                        info_str += f" {move_uci}:{prob[0]:.4f}"

                    return f"{info_str}\nbestmove {best_move.uci()}"
                else:
                    return f"bestmove {best_move.uci()}"
            except Exception as e:
                return f"info string Error generating move: {str(e)}\nbestmove a1a1"

        # Quit
        elif cmd == "quit":
            return "quit"

        # Unknown command
        return f"info string Unknown command: {command}"

    def run(self):
        """Run the UCI engine, reading from stdin and writing to stdout."""
        while True:
            try:
                command = input()
                response = self.process_command(command)

                if response:
                    print(response)

                if command.strip() == "quit":
                    break
            except EOFError:
                break
            except Exception as e:
                print(f"info string Error: {str(e)}")


def create_uci_engine(model_path=None, *, command_queue=None, response_queue=None):
    """Create and return a UCI-compatible engine that can be used with python-chess.

    This creates a subprocess that communicates via the UCI protocol.

    Args:
        model_path: Path to the model checkpoint
        command_queue: Queue for sending commands to the engine
        response_queue: Queue for receiving responses from the engine

    Returns:
        A chess.engine.SimpleEngine instance
    """
    # Build a simple wrapper script to launch the UCI engine
    script_content = f"""
        import os
        import sys
        sys.path.append(os.getcwd())
        from src.engines.fly_engine import UciFlyEngine

        model_path = "{model_path}" if "{model_path}" else None
        engine = UciFlyEngine(model_path)
        engine.run()
        """
    # Create a temporary script file
    script_path = os.path.join(os.getcwd(), "uci_fly_engine_wrapper.py")
    with open(script_path, "w") as f:
        f.write(script_content)

    # Launch the engine as a subprocess
    return chess.engine.SimpleEngine.popen_uci(command=["python", script_path])


if __name__ == "__main__":
    # If this file is run directly, start the UCI engine
    engine = UciFlyEngine()
    engine.run()
