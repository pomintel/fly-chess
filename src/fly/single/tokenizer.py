import numpy as np

# Removing TRANSFORMATION_DICT as we'll only support tensor format
KQ_LIST = ["K", "Q", "k", "q"]

# Piece type indices for each channel
WHITE_PAWN = 0
WHITE_KNIGHT = 1
WHITE_BISHOP = 2
WHITE_ROOK = 3
WHITE_QUEEN = 4
WHITE_KING = 5
BLACK_PAWN = 6
BLACK_KNIGHT = 7
BLACK_BISHOP = 8
BLACK_ROOK = 9
BLACK_QUEEN = 10
BLACK_KING = 11

# Piece to channel mapping
_PIECE_MAP = {
    "P": WHITE_PAWN,
    "N": WHITE_KNIGHT,
    "B": WHITE_BISHOP,
    "R": WHITE_ROOK,
    "Q": WHITE_QUEEN,
    "K": WHITE_KING,
    "p": BLACK_PAWN,
    "n": BLACK_KNIGHT,
    "b": BLACK_BISHOP,
    "r": BLACK_ROOK,
    "q": BLACK_QUEEN,
    "k": BLACK_KING,
}

# Channel indices in the final tensor
SIDE_TO_MOVE_CHANNEL = 12
CASTLING_K_CHANNEL = 13
CASTLING_Q_CHANNEL = 14
CASTLING_k_CHANNEL = 15
CASTLING_q_CHANNEL = 16
EN_PASSANT_CHANNEL = 17
HALFMOVE_CHANNEL = 18
FULLMOVE_CHANNEL = 19


def tokenize(fen):
    """
    Converts a FEN string into a 8x8x20 tensor representation:
    - 12 channels: 6 types of white pieces + 6 types of black pieces
    - 1 channel: side-to-move plane (1 for white, 0 for black)
    - 4 channels: KQkq castling rights, one channel each
    - 1 channel: en passant target square
    - 2 channels: halfmove and fullmove counts

    Args:
        fen: FEN string representation of chess position

    Returns:
        8x8x20 float tensor
    """
    parts = fen.split()
    if len(parts) < 6:
        raise ValueError("FEN must have 6 fields.")

    board_fen = parts[0]
    side = parts[1]
    castling_str = parts[2]  # e.g. "KQkq"
    en_passant_str = parts[3]  # e.g. "e3" or "-"
    halfmove_clock = int(
        parts[4]
    )  # Number of halfmoves since last capture or pawn advance
    fullmove_number = int(parts[5])  # Number of full moves, starts at 1

    # Initialize the output tensor with 20 channels
    tensor = np.zeros((8, 8, 20), dtype=float)

    # Parse the board representation
    ranks = board_fen.split("/")
    if len(ranks) != 8:
        raise ValueError("Board FEN must consist of 8 ranks, separated by '/'.")

    # Fill piece planes (first 12 channels)
    for row, rank_str in enumerate(ranks):
        col = 0
        for char in rank_str:
            if char.isdigit():
                steps = int(char)
                col += steps
            else:
                if char in _PIECE_MAP:
                    channel = _PIECE_MAP[char]
                    tensor[row, col, channel] = 1.0
                    col += 1
                else:
                    raise ValueError(f"Invalid piece character '{char}' in FEN.")
        if col != 8:
            raise ValueError(f"Rank '{rank_str}' does not describe exactly 8 columns.")

    # Side to move plane (channel 12)
    if side == "w":
        tensor[:, :, SIDE_TO_MOVE_CHANNEL] = 1.0
    # For black, leave as 0

    # Castling rights (channels 13-16)
    if "K" in castling_str:
        tensor[:, :, CASTLING_K_CHANNEL] = 1.0
    if "Q" in castling_str:
        tensor[:, :, CASTLING_Q_CHANNEL] = 1.0
    if "k" in castling_str:
        tensor[:, :, CASTLING_k_CHANNEL] = 1.0
    if "q" in castling_str:
        tensor[:, :, CASTLING_q_CHANNEL] = 1.0

    # En passant target square (channel 17)
    if en_passant_str != "-":
        file_char = en_passant_str[0]  # e.g. 'e'
        rank_char = en_passant_str[1]  # e.g. '3'
        file_idx = ord(file_char) - ord("a")  # 'a'->0, 'b'->1, etc.
        rank_idx = 8 - int(rank_char)  # '1'->7, '2'->6, etc.
        tensor[rank_idx, file_idx, EN_PASSANT_CHANNEL] = 1.0

    # Halfmove clock (channel 18) - normalized to [0, 1]
    # A common max value for halfmove clock is 100 (50-move rule)
    normalized_halfmove = min(halfmove_clock / 100.0, 1.0)
    tensor[:, :, HALFMOVE_CHANNEL] = normalized_halfmove

    # Fullmove number (channel 19) - normalized to [0, 1]
    # Typical games rarely go beyond 100 moves
    normalized_fullmove = min(fullmove_number / 100.0, 1.0)
    tensor[:, :, FULLMOVE_CHANNEL] = normalized_fullmove

    return tensor
