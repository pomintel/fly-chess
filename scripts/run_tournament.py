#!/usr/bin/env python
"""
Chess Engine Tournament Runner

Usage:
    python scripts/run_tournament.py [--num_games=N] [--engines=engine1,engine2,...]

Examples:
    python scripts/run_tournament.py --num_games=10
    python scripts/run_tournament.py --num_games=2 --engines=fly_uci,stockfish,random
"""

import sys
import os
import argparse
import subprocess
import signal
import datetime

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="Start a chess engine tournament")
    parser.add_argument(
        "--num_games",
        type=int,
        default=2,
        help="Number of games to play between each pair of engines",
    )
    parser.add_argument(
        "--engines",
        type=str,
        default="fly_uci,270M,stockfish,leela_chess_zero_depth_1,random",
        help="Comma-separated list of engines to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="tournament_results",
        help="Directory to save results",
    )
    return parser.parse_args()


def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}, exiting...")
    sys.exit(0)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"tournament_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    command = [
        sys.executable,
        "src/tournament.py",
        f"--num_games={args.num_games}",
    ]

    if args.engines:
        engines_list = args.engines.split(",")
        print(f"Using engines for tournament: {', '.join(engines_list)}")

        engines_str = ", ".join([f"'{engine}'" for engine in engines_list])
        print(f"Engine list: [{engines_str}]")

    print(f"Starting tournament with {args.num_games} games per engine pair")
    print(f"Results will be saved to: {output_dir}")
    print(f"Running command: {' '.join(command)}")

    try:
        # Run the process and directly connect stdin/stdout/stderr to the terminal
        process = subprocess.run(
            command,
            check=False,
        )

        if process.returncode == 0:
            print(f"\nTournament completed successfully!")
        else:
            print(f"\nError during tournament, return code: {process.returncode}")

    except Exception as e:
        print(f"Error running command: {e}")


if __name__ == "__main__":
    main()
