import os
import io
import chess
import chess.pgn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from typing import Optional, Dict, List, Tuple
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm
from src.chess.engines import engine as engine_lib
from src.chess.engines import constants

def evaluate_single_puzzle(args: Tuple[pd.Series, str]) -> Dict:
    """
    Evaluate a single puzzle in parallel processing.
    
    Args:
        args: Tuple containing (puzzle, agent_name)
        
    Returns:
        Dictionary with evaluation results
    """
    puzzle, agent_name = args
    # Create engine instance in the worker process
    engine = constants.ENGINE_BUILDERS[agent_name]()
    correct = evaluate_puzzle_from_pandas_row(puzzle=puzzle, engine=engine)
    
    return {
        'puzzle_id': puzzle.name,
        'correct': correct,
        'rating': puzzle['Rating']
    }

def evaluate_puzzle_from_pandas_row(puzzle: pd.Series, engine: engine_lib.Engine) -> bool:
    """
    Evaluate a single puzzle.
    
    Args:
        puzzle: Pandas Series containing puzzle data
        engine: Chess engine to evaluate
    
    Returns:
        True if puzzle was solved correctly, False otherwise
    """
    game = chess.pgn.read_game(io.StringIO(puzzle['PGN']))
    if game is None:
        raise ValueError(f'Failed to read game from PGN {puzzle["PGN"]}.')
    board = game.end().board()
    return evaluate_puzzle_from_board(
        board=board,
        moves=puzzle['Moves'].split(' '),
        engine=engine,
    )

def evaluate_puzzle_from_board(
    board: chess.Board,
    moves: List[str],
    engine: engine_lib.Engine,
) -> bool:
    """
    Returns True if the engine solves the puzzle and False otherwise.
    
    Args:
        board: Chess board position
        moves: List of moves in the puzzle
        engine: Chess engine to evaluate
        
    Returns:
        True if puzzle was solved correctly, False otherwise
    """
    for move_idx, move in enumerate(moves):
        # According to https://database.lichess.org/#puzzles, the FEN is the
        # position before the opponent makes their move. The position to present to
        # the player is after applying the first move to that FEN. The second move
        # is the beginning of the solution.
        if move_idx % 2 == 1:  # Engine's turn
            predicted_move = engine.play(board=board).uci()
            if move != predicted_move:
                board.push(chess.Move.from_uci(predicted_move))
                return board.is_checkmate()
        board.push(chess.Move.from_uci(move))
    return True

def evaluate_puzzles(engine: engine_lib.Engine, num_puzzles: Optional[int] = None, 
                    puzzles_path: Optional[str] = None, num_workers: Optional[int] = None,
                    agent_name: Optional[str] = None) -> Dict:
    """
    Evaluate engine performance on chess puzzles using parallel processing.
    
    Args:
        engine: The chess engine to evaluate
        num_puzzles: Number of puzzles to evaluate (None for all puzzles)
        puzzles_path: Path to puzzles CSV file (defaults to data/chess/puzzles.csv)
        num_workers: Number of parallel workers (defaults to CPU count - 1)
        agent_name: Name of the agent from ENGINE_BUILDERS
    
    Returns:
        Dictionary containing evaluation results
    """
    if puzzles_path is None:
        puzzles_path = os.path.join(os.getcwd(), 'data/chess/puzzles.csv')
    
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
        
    if agent_name is None:
        raise ValueError("agent_name must be provided")
    
    puzzles = pd.read_csv(puzzles_path, nrows=num_puzzles)
    total_puzzles = len(puzzles)
    
    # Use sequential processing for neural network engines
    is_neural_net = agent_name in {'9M', '136M', '270M'}
    
    if is_neural_net:
        print(f"\nEvaluating {total_puzzles} puzzles sequentially (neural network engine)...")
        results = []
        for _, puzzle in tqdm(puzzles.iterrows(), total=total_puzzles, desc="Evaluating puzzles", unit="puzzle"):
            result = {
                'puzzle_id': puzzle.name,
                'correct': evaluate_puzzle_from_pandas_row(puzzle=puzzle, engine=engine),
                'rating': puzzle['Rating']
            }
            results.append(result)
    else:
        print(f"\nEvaluating {total_puzzles} puzzles using {num_workers} workers...")
        # Prepare arguments for parallel processing - pass agent_name instead of engine instance
        puzzle_args = [(puzzle, agent_name) for _, puzzle in puzzles.iterrows()]
        
        # Run evaluation in parallel with progress bar
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(evaluate_single_puzzle, puzzle_args),
                total=total_puzzles,
                desc="Evaluating puzzles",
                unit="puzzle"
            ))
    
    # Group puzzles by rating intervals of 200
    rating_intervals = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for result in results:
        rating = result['rating']
        interval = f"{(rating // 200) * 200}-{((rating // 200) + 1) * 200}"
        
        rating_intervals[interval]['total'] += 1
        if result['correct']:
            rating_intervals[interval]['correct'] += 1
    
    return {
        'results': results,
        'rating_intervals': dict(rating_intervals)
    }

def save_evaluation_results(results: Dict, agent_name: str, output_dir: str = 'results'):
    """
    Save evaluation results to a pickle file.
    
    Args:
        results: Dictionary containing evaluation results
        agent_name: Name of the agent
        out_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{agent_name}_results.pkl')
    
    with open(file_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {file_path}")

def load_evaluation_results(agent_name: str, output_dir: str = 'results') -> Dict:
    """
    Load evaluation results from a pickle file.
    
    Args:
        agent_name: Name of the agent
        out_dir: Directory containing results
        
    Returns:
        Dictionary containing evaluation results
    """
    file_path = os.path.join(output_dir, f'{agent_name}_results.pkl')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No evaluation results found for {agent_name}")
    
    with open(file_path, 'rb') as f:
        results = pickle.load(f)
    return results

def plot_puzzle_results(results: Dict, out_path: str, puzzle_result_name: str):
    """
    Plot puzzle evaluation results.
    
    Args:
        results: Dictionary containing evaluation results
        out_path: Directory to save plot
        puzzle_result_name: Name prefix for the plot file
    """
    rating_intervals = results['rating_intervals']
    
    # Sort intervals by rating
    sorted_intervals = sorted(rating_intervals.items(), key=lambda x: int(x[0].split('-')[0]))
    
    # Calculate percentages and prepare plot data
    x_labels = []
    percentages = []
    total_correct = 0
    total_puzzles = 0
    
    for interval, stats in sorted_intervals:
        if stats['total'] > 0:
            percentage = stats['correct'] / stats['total']
            x_labels.append(f"{interval}\n{stats['total']}")
            percentages.append(percentage)
            total_correct += stats['correct']
            total_puzzles += stats['total']
    
    overall_acc = total_correct / total_puzzles if total_puzzles > 0 else 0
    
    # Create plot
    plt.figure(figsize=(15, 6))
    x_positions = np.arange(len(x_labels))
    plt.bar(x_positions, percentages, width=0.5)
    
    plt.xlabel('Puzzle Rating (Elo) - Count')
    plt.ylabel('Accuracy (%)')
    plt.ylim([0.0, 1.0])
    plt.yticks(np.arange(0.0, 1.05, 0.05))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plot_title = f"Percentage of Correct Results per Rating Interval\n{puzzle_result_name}\nAcc = {overall_acc:.4f}"
    plt.title(plot_title)
    plt.xticks(x_positions, x_labels, rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(out_path, f"{puzzle_result_name}_rating_histogram.png"))
    plt.close() 