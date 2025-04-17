import numpy as np

TRANSFORMATION_DICT = {"basicrnn": "vector", "basicCNN": "tensor"}
KQ_LIST = ["K", "Q", "k", "q"]

_PIECE_MAP = {
    "P": (0, 1),
    "p": (0, -1),
    "N": (1, 1),
    "n": (1, -1),
    "B": (2, 1),
    "b": (2, -1),
    "R": (3, 1),
    "r": (3, -1),
    "Q": (4, 1),
    "q": (4, -1),
    "K": (5, 1),
    "k": (5, -1),
}


def tokenize(fen):
    transform = TRANSFORMATION_DICT["basicCNN"]

    parts = fen.split()
    if len(parts) < 4:
        raise ValueError("FEN must have at least 4 fields.")
    board_fen = parts[0]
    side = parts[1]
    castling_str = parts[2]  # e.g. "KQkq"
    en_passant_str = parts[3]  # e.g. "e3" or "-"

    piece_array = np.zeros((8, 8, 6)).astype(int)
    KQ_dict = {}

    ranks = board_fen.split("/")
    if len(ranks) != 8:
        raise ValueError("Board FEN must consist of 8 ranks, separated by '/'.")

    for row, rank_str in enumerate(ranks):
        col = 0
        for char in rank_str:
            if char.isdigit():
                steps = int(char)
                col += steps
            else:
                if char in _PIECE_MAP:
                    piece_idx, sign = _PIECE_MAP[char]
                    piece_array[row, col, piece_idx] = sign
                    if char in KQ_LIST:
                        KQ_dict[char] = [row, col]
                    col += 1
                else:
                    raise ValueError(f"Invalid piece character '{char}' in FEN.")
        if col != 8:
            raise ValueError(f"Rank '{rank_str}' does not describe exactly 8 columns.")

    if side == "b":
        piece_array = piece_array * -1

    if transform == "tensor":
        castling_plane = np.zeros((8, 8)).astype(int)
        for piece in KQ_dict:
            if piece in castling_str:
                castling_plane[KQ_dict[piece][0], KQ_dict[piece][1]] = (
                    1  # this should be okay, they are still at the original places
                )

        enpass_plane = np.zeros((8, 8)).astype(int)
        if en_passant_str != "-":
            file_char = en_passant_str[0]  # e.g. 'e'
            rank_char = en_passant_str[1]  # e.g. '3'
            file_idx = ord(file_char) - ord("a")  # 'a'->0, 'b'->1,
            rank_idx = 8 - int(rank_char)  # '1'->7, '2'->6,
            enpass_plane[rank_idx, file_idx] = 1

        castling_plane_3d = castling_plane[:, :, np.newaxis]  # (8,8,1)
        enpass_plane_3d = enpass_plane[:, :, np.newaxis]  # (8,8,1)
        out_tensor = np.concatenate(
            [piece_array, castling_plane_3d, enpass_plane_3d], axis=2
        )  # (8,8,8)
        return out_tensor.astype(float)

    elif transform == "vector":
        flat_pieces = piece_array.flatten()  # shape (8*8*6,) = (384,)

        # Think of it as it is tied to the location, probably not swapping case...
        castling_vec = np.zeros(4).astype(int)
        for char in KQ_LIST:
            if char in castling_str:
                castling_vec[KQ_LIST.index(char)] = 1

        enpass_file_ohe = np.zeros((2, 8)).astype(int)
        if en_passant_str != "-":
            file_idx = ord(en_passant_str[0]) - ord("a")
            rank_temp = int(en_passant_str[1])
            if rank_temp == 3:
                rank_idx = 0
            elif rank_temp == 6:
                rank_idx = 1
            else:
                raise ValueError("Invalid en passant location!")
            enpass_file_ohe[rank_idx, file_idx] = 1
        enpass_file_ohe = enpass_file_ohe.reshape(
            16,
        )

        out_vector = np.concatenate([flat_pieces, castling_vec, enpass_file_ohe])
        return out_vector.astype(float)

    else:
        raise ValueError("transform must be either 'tensor' or 'vector'.")
