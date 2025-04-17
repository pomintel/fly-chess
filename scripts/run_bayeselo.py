import subprocess

PGN_PATH = "./data/chess/tournament_games.pgn"
BAYESELO_EXEC = "./bayeselo/bayeselo"

commands = f"""
readpgn {PGN_PATH}
elo
mm
ratings
"""


def run_bayeselo():
    print("Running BayesElo...\n")
    result = subprocess.run(
        [BAYESELO_EXEC], input=commands, capture_output=True, text=True
    )

    if result.returncode != 0:
        print("BayesElo failed:", result.stderr)
        return

    output = result.stdout
    if "Rank Name" in output:
        rating_block = output.split("Rank Name", 1)[1]
        print("Rank Name" + rating_block)
    else:
        print("BayesElo output:\n", output)


if __name__ == "__main__":
    run_bayeselo()
