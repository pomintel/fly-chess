#!/usr/bin/env python3
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
        original_board = board.copy()

        all_move_data = []
        legal_moves = list(board.legal_moves)
        move_probs = {}

        for move in legal_moves:
            board.push(move)
            data = tokenize(board.fen())
            data = torch.from_numpy(data).float()
            data = data.to(self.device)
            all_move_data.append(data)
            board.pop()

        if all_move_data:
            data = torch.stack(all_move_data, dim=0)
            output = self.action_chooser.model(data)
            probs = F.softmax(output, dim=0)

            for i, move in enumerate(legal_moves):
                move_probs[move.uci()] = probs[i].tolist()

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
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        if model_path is None:
            model_path = os.path.join(
                project_root,
                "results/DPU_CNN_Unlearnable_1filters_2560000_trial1_2Timesteps-signed/model.pth",
            )
        elif not os.path.isabs(model_path):
            model_path = os.path.join(project_root, model_path)

        self.engine = FlyEngine(model_path)
        self.board = chess.Board()
        self.debug = False
        self.name = "FlyChess"
        self.author = "Pomintel"
        self.move_overhead = 30  # Default move overhead in milliseconds
        self.threads = 1  # Default number of threads
        self.hash_size = 16  # Default hash size in MB
        self.syzygy_path = ""  # Default empty path for Syzygy endgame tablebases
        self.show_wdl = True  # Default to showing WDL

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
                "option name Move Overhead type spin default 30 min 0 max 5000\n"
                "option name Threads type spin default 1 min 1 max 512\n"
                "option name Hash type spin default 16 min 1 max 33554432\n"
                "option name SyzygyPath type string default \n"
                "option name UCI_ShowWDL type check default true\n"
                "option name Model Path type string default results/DPU_CNN_Unlearnable_1filters_2560000_trial1_2Timesteps-signed/model.pth\n"
                "uciok"
            )

        elif cmd == "isready":
            return "readyok"

        elif cmd == "setoption":
            if len(tokens) >= 5 and tokens[1] == "name" and tokens[3] == "value":
                option_name = " ".join(tokens[2 : tokens.index("value")])
                option_value = tokens[4]

                if option_name == "Debug":
                    self.debug = option_value.lower() == "true"
                elif option_name == "Move Overhead":
                    try:
                        self.move_overhead = int(option_value)
                    except ValueError:
                        pass  # Ignore invalid values
                elif option_name == "Threads":
                    try:
                        self.threads = int(option_value)
                        # We don't actually use multiple threads, but we accept the option
                    except ValueError:
                        pass  # Ignore invalid values
                elif option_name == "Hash":
                    try:
                        self.hash_size = int(option_value)
                        # We don't actually use hash tables, but we accept the option
                    except ValueError:
                        pass  # Ignore invalid values
                elif option_name == "SyzygyPath":
                    self.syzygy_path = option_value
                elif option_name == "UCI_ShowWDL":
                    self.show_wdl = option_value.lower() == "true"
                elif option_name == "Model Path":
                    # Get project root path
                    project_root = os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    )
                    # Make path absolute if it's not already
                    if not os.path.isabs(option_value):
                        option_value = os.path.join(project_root, option_value)
                    # Create a new engine instance with the specified model
                    self.engine = FlyEngine(option_value)

            return ""

        elif cmd == "ucinewgame":
            self.board = chess.Board()
            return ""

        elif cmd == "position":
            if len(tokens) < 2:
                return ""

            if tokens[1] == "startpos":
                self.board = chess.Board()
                move_idx = 3 if len(tokens) > 2 and tokens[2] == "moves" else 0
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
            # Handle time control parameters
            wtime = btime = winc = binc = None
            movetime = None
            depth = None

            # Parse parameters
            i = 1
            while i < len(tokens):
                if tokens[i] == "wtime" and i + 1 < len(tokens):
                    wtime = int(tokens[i + 1])
                    i += 2
                elif tokens[i] == "btime" and i + 1 < len(tokens):
                    btime = int(tokens[i + 1])
                    i += 2
                elif tokens[i] == "winc" and i + 1 < len(tokens):
                    winc = int(tokens[i + 1])
                    i += 2
                elif tokens[i] == "binc" and i + 1 < len(tokens):
                    binc = int(tokens[i + 1])
                    i += 2
                elif tokens[i] == "movetime" and i + 1 < len(tokens):
                    movetime = int(tokens[i + 1])
                    i += 2
                elif tokens[i] == "depth" and i + 1 < len(tokens):
                    depth = int(tokens[i + 1])
                    i += 2
                else:
                    i += 1

            # Calculate time to think (Lichess will handle this for us, but it's good to have)
            # This doesn't actually affect our engine's thinking time but demonstrates correct handling
            think_time = None
            if movetime is not None:
                # If movetime is specified, use that minus the move overhead
                think_time = max(1, movetime - self.move_overhead)
            elif wtime is not None and btime is not None:
                # Calculate based on whose turn it is
                if self.board.turn == chess.WHITE and wtime is not None:
                    remaining = wtime
                    increment = winc if winc is not None else 0
                else:
                    remaining = btime
                    increment = binc if binc is not None else 0

                # Simple time management: use 1/20th of remaining time + half of increment
                think_time = remaining // 20 + increment // 2 - self.move_overhead
                think_time = max(1, think_time)  # Ensure positive think time

            try:
                analysis = self.engine.analyse(self.board)
                move_probs = analysis.get("move_probs", {})

                best_move = self.engine.play(self.board)

                score_value = 0
                for move_uci, prob in move_probs.items():
                    if move_uci == best_move.uci():
                        score_value = int(prob[0] * 100)
                        break

                if self.debug:
                    info_lines = []
                    for move_uci, prob in sorted(
                        move_probs.items(), key=lambda x: x[1][0], reverse=True
                    ):
                        info_lines.append(f"info string {move_uci}:{prob[0]:.4f}")

                    if think_time is not None:
                        info_lines.append(
                            f"info string think_time={think_time}ms overhead={self.move_overhead}ms"
                        )

                    info_response = "\n".join(info_lines)

                    info_response += (
                        f"\ninfo score cp {score_value} depth 1 pv {best_move.uci()}"
                    )

                    if self.show_wdl:
                        # Convert centipawn score to win/draw/loss probabilities
                        # This is a simple model - in a real engine this would be more sophisticated
                        wdl_w = min(
                            1000, max(0, 500 + score_value // 2)
                        )  # Win probability (0-1000)
                        wdl_l = 1000 - wdl_w  # Loss probability
                        wdl_d = 0  # Draw probability - set to 0 for simplicity
                        info_response += f" wdl {wdl_w} {wdl_d} {wdl_l}"

                    return info_response + f"\nbestmove {best_move.uci()}"
                else:
                    info_response = (
                        f"info score cp {score_value} depth 1 pv {best_move.uci()}"
                    )

                    if self.show_wdl:
                        # Convert centipawn score to win/draw/loss probabilities
                        wdl_w = min(1000, max(0, 500 + score_value // 2))
                        wdl_l = 1000 - wdl_w
                        wdl_d = 0
                        info_response += f" wdl {wdl_w} {wdl_d} {wdl_l}"

                    return info_response + f"\nbestmove {best_move.uci()}"
            except Exception as e:
                return f"info string Error generating move: {str(e)}\nbestmove a1a1"

        elif cmd == "quit":
            return "quit"

        elif cmd == "stop":
            try:
                best_move = self.engine.play(self.board)
                return f"bestmove {best_move.uci()}"
            except:
                return "bestmove a1a1"

        return ""

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
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    if model_path and not os.path.isabs(model_path):
        model_path = os.path.join(project_root, model_path)

    script_content = f"""
    import os
    import sys
    
    project_root = "{project_root}"
    sys.path.append(project_root)
    
    from src.engines.fly_engine import UciFlyEngine

    model_path = "{model_path}" if "{model_path}" else None
    engine = UciFlyEngine(model_path)
    engine.run()
    """
    script_path = os.path.join(project_root, "uci_fly_engine_wrapper.py")
    with open(script_path, "w") as f:
        f.write(script_content)

    return chess.engine.SimpleEngine.popen_uci(command=["python", script_path])


if __name__ == "__main__":
    engine = UciFlyEngine()
    engine.run()
