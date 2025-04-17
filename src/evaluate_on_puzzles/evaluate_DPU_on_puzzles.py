from collections.abc import Sequence
import io
import os
import chess
import chess.engine
import chess.pgn
import numpy as np
from tqdm import tqdm
import pandas as pd

from src.evaluate_on_puzzles.plotting import plot_puzzle_results
from src.model.action_chooser import ActionChooser

def evaluate_puzzle_from_pandas_row(
    puzzle,
    engine
):
  """Returns True if the `engine` solves the puzzle and False otherwise."""
  game = chess.pgn.read_game(io.StringIO(puzzle['PGN']))
  if game is None:
    raise ValueError(f'Failed to read game from PGN {puzzle["PGN"]}.')
  board = game.end().board()
  moves = puzzle['Moves'].split(' ')
  return (len(moves),
          evaluate_puzzle_from_board(
            board=board,
            moves = moves,
            engine=engine,
          ))


def evaluate_puzzle_from_board(
    board: chess.Board,
    moves: Sequence[str],
    engine
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


def eval_DPU_on_puzzles(out_path,score_filter = None):

  result_df = pd.DataFrame(columns=[f"{i}-{i + 200}" for i in range(200, 3000, 200)])
  result_df.index.name = 'puzzle_len'

  puzzles_path = os.path.join(
      os.getcwd(),
      'data/chess_data/puzzles.csv',
  )
  puzzles = pd.read_csv(puzzles_path)
  if score_filter is not None:
    puzzles = puzzles[puzzles['Rating'] < score_filter]
    puzzle_result_name = f"puzzle_result_{str(score_filter)}"
  else:
    puzzle_result_name = f"puzzle_result"
  engine = ActionChooser(os.path.join(out_path, 'model.pth'))

  for puzzle_id, puzzle in tqdm(puzzles.iterrows(), total=len(puzzles), desc="Evaluating puzzles"):
    puzzle_len, correct = evaluate_puzzle_from_pandas_row(
      puzzle=puzzle,
      engine=engine,
    )
    interval = get_interval_str(puzzle['Rating'])
    if puzzle_len not in result_df.index:
      result_df.loc[puzzle_len] = {col: np.array([0,0]) for col in result_df.columns}
    result_df.loc[puzzle_len, interval] = result_df.loc[puzzle_len, interval] + [correct,1]
  result_df.to_pickle(os.path.join(out_path, puzzle_result_name + '.pkl'))

  # plot and save barplot
  plot_puzzle_results(result_df, out_path, puzzle_result_name)

  return result_df

