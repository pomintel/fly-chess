#!/usr/bin/env python3

import os
import argparse
from src.chess.evaluate import evaluate_puzzles, plot_puzzle_results, save_evaluation_results, load_evaluation_results
from src.chess.engines import constants

def main():
    parser = argparse.ArgumentParser(description='Evaluate chess engine on puzzles')
    parser.add_argument('--agent', type=str, required=True,
                       choices=['local','9M','136M','270M','stockfish', 'stockfish_all_moves', 
                               'leela_chess_zero_depth_1',
                               'leela_chess_zero_policy_net',
                               'leela_chess_zero_400_sims'],
                       help='The chess engine to evaluate')
    parser.add_argument('--num_puzzles', type=int, default=None,
                       help='Number of puzzles to evaluate (default: all)')
    parser.add_argument('--puzzles_path', type=str, 
                       default=os.path.join(os.getcwd(), 'data/chess/puzzles.csv'),
                       help='Path to puzzles CSV file')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--load_existing', action='store_true', help='Load existing results instead of evaluating')
    
    args = parser.parse_args()
    
    if args.agent not in constants.ENGINE_BUILDERS:
        raise ValueError(f"Unknown agent: {args.agent}. Available agents: {list(constants.ENGINE_BUILDERS.keys())}")

    if args.load_existing:
        print(f"Loading existing results for {args.agent}...")
        results = load_evaluation_results(args.agent, args.output_dir)
    else:
        print(f"Evaluating {args.agent} on puzzles...")
        engine = constants.ENGINE_BUILDERS[args.agent]()
        results = evaluate_puzzles(
            engine=engine,
            num_puzzles=args.num_puzzles,
            puzzles_path=args.puzzles_path,
            num_workers=args.num_workers,
            agent_name=args.agent
        )
        save_evaluation_results(results, args.agent, args.output_dir)

    # Plot results
    os.makedirs(args.output_dir, exist_ok=True)
    plot_puzzle_results(results, args.output_dir, args.agent)
    
    # Print overall accuracy
    total_correct = sum(r['correct'] for r in results['results'])
    total_puzzles = len(results['results'])
    print("\nEvaluation Results:")
    print(f"Overall accuracy: {total_correct/total_puzzles:.4f}")
    print(f"Total puzzles evaluated: {total_puzzles}")
    print(f"Workers used: {args.num_workers if args.num_workers else max(1, os.cpu_count() - 1)}")
    print(f"Results saved to: {os.path.join(args.output_dir, f'{args.agent}_rating_histogram.png')}")

if __name__ == '__main__':
    main() 