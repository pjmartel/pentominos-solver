#!/usr/bin/env python
#
# Pentominos solver
#
#                  --PJM2019
import argparse
import numpy as np
from scipy.signal import convolve2d
from sys import exit

# Pentominos is a list of triplets
# 0 - free pentomino shape (numpy 3x3 array)
# 1 - Number of symmetric images  (max 2)
# 2 - Number of rotations (max 4)
pentominos = [
    [np.array([[0, 1, 0],        # 1 , "X"
               [1, 1, 1],
               [0, 1, 0]]), 1, 1],

    [np.array([[1, 1, 1],        # 2 , "V"
               [1, 0, 0],
               [1, 0, 0]]), 1, 4],

    [np.array([[1, 1, 0],        # 3 , "U"
               [1, 0, 0],
               [1, 1, 0]]), 1, 4],

    [np.array([[1, 1, 1],        # 4 , "T"
               [0, 1, 0],
               [0, 1, 0]]), 1, 4],

    [np.array([[1, 1, 0],        # 5 , "W"
               [0, 1, 1],
               [0, 0, 1]]), 1, 4],

    [np.array([[1, 1, 0],        # 6 , "Z"
               [0, 1, 0],
               [0, 1, 1]]), 2, 2],

    [np.array([[1, 0, 0],        # 7 , "P"
               [1, 1, 0],
               [1, 1, 0]]), 2, 4],

    [np.array([[1, 1, 0],        # 8 , "F"
               [0, 1, 1],
               [0, 1, 0]]), 2, 4],

    [np.array([[0, 0, 1, 0, 0],  # 9 , "I"
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0]]), 1, 2],

    [np.array([[0, 0, 0, 0, 0],  # 10 , "L"
               [0, 0, 1, 1, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0]]), 2, 4],

    [np.array([[0, 0, 0, 0, 0],  # 11 , "Y"
               [0, 0, 1, 0, 0],
               [0, 0, 1, 1, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0]]), 2, 4],

    [np.array([[0, 0, 0, 0, 0],  # 12 , "N"
               [0, 0, 0, 1, 0],
               [0, 0, 1, 1, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0]]), 2, 4]
]


# Recursively finds contiguous
# empty areas in boards
# the paramenter c returns the total
# square count of contiguous area
def flood_fill(b, i, j, c):
    """
Counts squares in empty (0) areas using
a recursive flood fill algorithm.
"""
    if b[i, j] != 0:
        return c
    c += 1
    b[i, j] = -1
    c = flood_fill(b, i+1,   j, c)
    c = flood_fill(b, i, j+1, c)
    c = flood_fill(b, i-1,   j, c)
    c = flood_fill(b, i,   j-1, c)
    return c


def valid_space(b):
    """
Check if empty board areas are multiples of 5.
Returns False if A % 5 != for any empty area A,
otherwise True.
    """

    for i in range(2, b.shape[0]-2):
        for j in range(2, b.shape[1]-2):
            c = flood_fill(b, i, j, c=0)
            if (c % 5) != 0:
                b[np.where(b == -1)] = 0
                return False
    b[np.where(b == -1)] = 0
    return True


def tile_board(p, b, c):
    """
Recursively tile board with backtracking.
    """
    global solveAll

    if(c == 0):
        print("Found a solution:")
        print()
        print(b[2:-2, 2:-2])
        print()
        if solveAll:
            return
        else:
            exit(0)

    bv = b[2:-2, 2:-2]

    piece = p[len(p)-c]
    pieceCode = len(p)+1-c
    for i in range(piece[1]):  # mirror
        for j in range(piece[2]):  # rotation
            # Find list of valid piece positions with a 2D convolution
            # NB: the convolution kernel must be reversed (::-1,::-1)
            C = convolve2d(bv, piece[0][::-1, ::-1], 'same', fillvalue=1)
            freePositions = np.where(C == 0)
            # if(C.min() > 0): # no free positions for this piece
            if freePositions[0].size == 0:  # no free positions for piece
                piece[0] = np.rot90(piece[0])  # continue skips final rotation
                continue

            # Explore freely available positions
            hw_i = (piece[0].shape[0]-1)//2
            hw_j = (piece[0].shape[1]-1)//2
            flag = piece[0].astype(bool)
            #
            for i in range(freePositions[0].size):
                ioff, joff = freePositions[0][i]+2, freePositions[1][i]+2

                # Place piece on the board
                b[ioff-hw_i:ioff+hw_i+1, joff -
                  hw_j:joff+hw_j+1][flag] = pieceCode

                # Check for empty areas not multiples of 5 squares
                if valid_space(b):
                    tile_board(p, b, c-1)  # call fill_board for next piece

                # Remove piece from the board
                b[ioff-hw_i:ioff+hw_i+1, joff -
                  hw_j:joff+hw_j+1][flag] = 0

            # No positions available for this piece orientation
            # Try next piece rotation
            piece[0] = np.rot90(piece[0])
        # Try mirrored version of piece
        piece[0] = np.flip(piece[0], axis=1)
    return


# Parse commmand line arguments
parser = argparse.ArgumentParser(
    "psolver.py", description="Tiles rectangular boards with sets of pentominos.")
parser.add_argument(
    "-A", "--all", help="find all tilings for current board", action="store_true")
parser.add_argument("-B", "--board", nargs=2, metavar=('rows', 'cols'),
                    help="Size of tiling board (default: 6 x 10)", type=int, default=[6, 10])
parser.add_argument("-P", "--pieces", nargs="+",
                    help="list of pentominos to use (in search order)", metavar="piece", type=int,
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

args = parser.parse_args()
rows = args.board[0]
cols = args.board[1]
solveAll = args.all
order = args.pieces

if not set(args.pieces).issubset(range(1, 13)):
    print("Error: Pentomino pieces must be in the range 1-12")
    exit(0)

# Ready to go
print("Finding a tiling for a {} x {} board.".format(rows, cols))
print("Using the following pieces: ", ', '.join(map(str, order)))
if(solveAll):
    print("Finding all tilings for a {} x {} board.".format(rows, cols))
    print("Warning: this may take a very long a time.")
print("")


# Pentominos list contains the selected pieces
pentominos = [pentominos[i-1] for i in order]

# Initialize board
board = np.zeros((rows+4, cols+4), dtype=np.int8)

# Frame the board with 2 layers of value "1" cells
board[:, [0, 1, -2, -1]] = 1
board[[0, 1, -2, -1], :] = 1

# The inner board
bv = board[2:-2, 2:-2]

# Init the position tracking array
pos = []

# Call tile_board for top piece
tile_board(pentominos, board, len(pentominos))

# If we get here, the board has not been fully tiled
# (eg. a non-multiple of 5 rectangle)
# or all solutions were found
if(not solveAll):
    print("Failed to tile board with the current set of pieces")
else:
    print("Found all solutions for the given board and pieces")
    exit(0)
