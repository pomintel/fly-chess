from collections.abc import Sequence
import io
import os
import chess
import chess.engine
import chess.pgn
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
from typing import Dict

from src.engines import fly_engine


def evaluate_puzzle_from_pandas_row(puzzle, engine):
    """Returns True if the `engine` solves the puzzle and False otherwise."""
    game = chess.pgn.read_game(io.StringIO(puzzle["PGN"]))
    if game is None:
        raise ValueError(f'Failed to read game from PGN {puzzle["PGN"]}.')
    board = game.end().board()
    moves = puzzle["Moves"].split(" ")
    return (
        len(moves),
        evaluate_puzzle_from_board(
            board=board,
            moves=moves,
            engine=engine,
        ),
    )


def evaluate_puzzle_from_board(
    board: chess.Board, moves: Sequence[str], engine
) -> bool:
    """Returns True if the `engine` solves the puzzle and False otherwise."""
    for move_idx, move in enumerate(moves):
        if move_idx % 2 == 1:
            predicted_move = engine.play(board).uci()
            if move != predicted_move:
                board.push(chess.Move.from_uci(predicted_move))
                return board.is_checkmate()
        board.push(chess.Move.from_uci(move))
    return True


def get_interval_str(value, start=200, end=3000, step=200):
    if value < start or value >= end:
        raise ValueError(f"Input {value} is out of range.")
    bucket = (value - start) // step
    lower = start + bucket * step
    return f"{lower}-{lower + step}"


def save_evaluation_results(results: Dict, agent_name: str, out_dir: str):
    """
    Save evaluation results to a pickle file.

    Args:
        results: Dictionary containing evaluation results
        agent_name: Name of the agent
        out_dir: Directory to save results
    """
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, f"{agent_name}_results.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {file_path}")


def load_evaluation_results(agent_name: str, out_dir: str) -> Dict:
    """
    Load evaluation results from a pickle file.

    Args:
        agent_name: Name of the agent
        out_dir: Directory containing results

    Returns:
        Dictionary containing evaluation results
    """
    file_path = os.path.join(out_dir, f"{agent_name}_results.pkl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No evaluation results found for {agent_name}")

    with open(file_path, "rb") as f:
        results = pickle.load(f)
    return results


def eval_DPU_on_puzzles(out_path, score_filter=None):
    result_df = pd.DataFrame(columns=[f"{i}-{i + 200}" for i in range(200, 3000, 200)])
    result_df.index.name = "puzzle_len"

    puzzles_path = os.path.join(
        os.getcwd(),
        "data/chess/puzzles.csv",
    )
    puzzles = pd.read_csv(puzzles_path)
    if score_filter is not None:
        puzzles = puzzles[puzzles["Rating"] < score_filter]
        puzzle_result_name = f"puzzle_result_{str(score_filter)}"
    else:
        puzzle_result_name = "puzzle_result"

    # Use FlyChessEngine which implements the chess.engines.Engine interface
    engine = fly_engine.FlyEngine(os.path.join(out_path, "model.pth"))

    for puzzle_id, puzzle in tqdm(
        puzzles.iterrows(), total=len(puzzles), desc="Evaluating puzzles"
    ):
        puzzle_len, correct = evaluate_puzzle_from_pandas_row(
            puzzle=puzzle,
            engine=engine,
        )
        interval = get_interval_str(puzzle["Rating"])
        if puzzle_len not in result_df.index:
            result_df.loc[puzzle_len] = {
                col: np.array([0, 0]) for col in result_df.columns
            }
        result_df.loc[puzzle_len, interval] = result_df.loc[puzzle_len, interval] + [
            correct,
            1,
        ]
    result_df.to_pickle(os.path.join(out_path, puzzle_result_name + ".pkl"))

    # plot and save barplot
    plot_puzzle_results(result_df, out_path, puzzle_result_name)

    return result_df


CMAP = mcolors.LinearSegmentedColormap.from_list("WhiteRed", ["white", "red"])


def plot_bar(
    result_df, out_path=None, puzzle_result_name=None, choice="elo", no_plot=False
):
    """
    choice = 'elo' or 'puzzle_len'
    """
    if no_plot == False:
        assert out_path is not None
        assert puzzle_result_name is not None

    df = result_df.copy()
    total_counts = []
    correct_counts = []

    if choice == "elo":
        sorted_x_label = sorted(df.columns, key=lambda x: int(x.split("-")[0]))
        xlabel = "Puzzle Rating (Elo) - Count"
        title_name = "Rating Interval"
        save_name = "_Elo_bar.png"
    elif choice == "puzzle_len":
        df = df.T
        sorted_x_label = sorted(df.columns)
        xlabel = "Puzzle Length - Count"
        title_name = "Puzzle Length"
        save_name = "_puzzlelen_bar.png"
    else:
        raise ValueError("Invalid selection")

    for name in sorted_x_label:
        col_totals = df[name].apply(lambda cell: cell[1])
        col_corrects = df[name].apply(lambda cell: cell[0])
        correct, total = col_corrects.sum(), col_totals.sum()
        total_counts.append(total)
        correct_counts.append(correct)
    percentages = [
        (corr / tot) if tot != 0 else 0.0
        for corr, tot in zip(correct_counts, total_counts)
    ]
    overall_acc = np.sum(correct_counts) / np.sum(total_counts)

    x_labels = [f"{name}\n{tot}" for name, tot in zip(sorted_x_label, total_counts)]
    x_positions = np.arange(len(sorted_x_label))

    if no_plot:
        return x_labels, percentages, overall_acc

    plt.figure(figsize=(15, 6))
    plt.bar(x_positions, percentages)
    plt.xlabel(xlabel)
    plt.ylabel("Accuracy (%)")
    plt.ylim([0.0, 1.0])
    plt.yticks(np.arange(0.0, 1.05, 0.05))
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plot_title = f"Percentage of Correct Results per {title_name}\n {out_path.split('/')[-1]}\n Acc = {overall_acc:.4f}"
    plt.title(plot_title)
    plt.xticks(x_positions, x_labels)
    plt.tight_layout()

    plt.savefig(os.path.join(out_path, puzzle_result_name + save_name))
    plt.close()


def plot_heatmap(result_df, out_path, puzzle_result_name):
    df = result_df.copy()
    total_counts = 0.0
    correct_counts = 0.0
    for idx in df.index:
        for col in df.columns:
            cell_val = df.loc[idx, col]
            corr, tot = cell_val
            total_counts += tot
            correct_counts += corr
            if tot == 0:
                pct = np.nan
            else:
                pct = corr / tot
            df.loc[idx, col] = pct
    overall_acc = correct_counts / total_counts
    df = df.astype(float)
    df = df.sort_index()

    masked_array = np.ma.masked_invalid(df.values)
    cmap_mod = plt.get_cmap(CMAP).copy()
    cmap_mod.set_bad(color="lightgrey")

    plt.figure(figsize=(10, 6))
    plt.imshow(
        masked_array,
        aspect="auto",
        cmap=cmap_mod,
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    plt.colorbar(label="% Correct")
    plt.xticks(ticks=np.arange(len(df.columns)), labels=df.columns, rotation=45)
    plt.yticks(ticks=np.arange(len(df.index)), labels=df.index)
    plt.xlabel("Puzzle Range")
    plt.ylabel("Puzzle Length")
    plot_title = f"Heatmap of Correct Results Percentage\n {out_path.split('/')[-1]}\n Acc = {overall_acc:.4f}"
    plt.title(plot_title)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, puzzle_result_name + "_heatmap.png"))
    plt.close()


def plot_puzzle_results(result_df, out_path, puzzle_result_name):
    plot_heatmap(result_df, out_path, puzzle_result_name)
    plot_bar(result_df, out_path, puzzle_result_name, choice="elo")
    plot_bar(result_df, out_path, puzzle_result_name, choice="puzzle_len")
