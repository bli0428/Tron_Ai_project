from types import new_class
import numpy as np
from trontypes import CellType

cell_conversion = {
    CellType.WALL: -2,
    CellType.BARRIER: -1,
    CellType.SPACE: 1,
    CellType.TRAP: 2,
    CellType.SPEED: 4,
    CellType.ARMOR: 8,
    CellType.BOMB: 16,
}

def get_converted_boards(board, pi, player, winner):
    """
    Takes in a Tron board and converts all strings into numbers. In addition, puts each
    board in the perspective of same player.

    Returns a list of tuples of symmetrical boards with corresponding policies, and whether the player who
    corresponds to the boards is the winner
    """
    canonical_board = convert_board(board, pi, player,)
    boards = get_board_symmetries(canonical_board, pi)
    output = []
    for b in boards:
        output.append((pad_board(b[0]),b[1],player==winner))
    
def convert_board(board, player):
    # Determining what to transform player into requires a little extra logic, players will be represented
    # by -32 and 32

    p = 1 if player == 0 else -1

    board_width = len(board)
    board_height = len(board[0])
    canonical_board = np.ones((board_height,board_width))
    for i in range(board_width):
        for j in range(board_height):
            cell = board[i][j]
            if cell == "1":
                canonical_board[i,j] = 32 * p
            elif cell == "2":
                canonical_board[i,j] = -32 * p
            else:
                canonical_board[i,j] = cell_conversion[cell]
    return canonical_board

def pad_board(board, width=17, height=17):
    # Need to make all boards 17x17 so that we can treat all boards the same in CNN
    # Will do this by making all additional spaces as unreachable walls
    final_board = np.ones((height,width)) * cell_conversion[CellType.WALL]
    board_width = board.shape[1]
    board_height = board.shape[0]
    final_board[:board_height,:board_width] = board
    return final_board


def get_board_symmetries(board, pi):
    """
    Takes in a canonical board representation (numerical representation from the current player's perspective)
    and returns all symmetries.
    Rotation is a little weird for policies because we need to stay faithful to the [U,D,L,R] order
    """
    l = []
    pi = np.array(pi)

    # 0 -> [U,D,L,R]    # 0R-> [U,D,R,L]
    # 1 -> [R,L,U,D]    # 1R-> [R,L,D,U]
    # 2 -> [D,U,R,L]    # 2R-> [D,U,L,R]
    # 3 -> [L,R,D,U]    # 3R-> [L,R,U,D]
    
    for i in range(1, 5):
        for j in [True, False]:
            new_b = np.rot90(board, i)
            new_pi = pi[[3,2,0,1]]
            for _ in range(i-1):
                new_pi = new_pi[[3,2,0,1]]
            if j:
                new_b = np.fliplr(new_b)
                new_pi = new_pi[[0,1,3,2]]
        l += [(new_b, new_pi)]
    return l