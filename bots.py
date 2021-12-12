#!/usr/bin/python

import numpy as np
from numpy.lib.polynomial import polysub
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math
from queue import Queue, PriorityQueue
from typing import Any
from dataclasses import dataclass, field
from agent import Net
from mcts import MonteCarloSearchTree
import time
# Throughout this file, ASP means adversarial search problem.

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)
class StudentBot:
    """ Write your student bot here"""
    def __init__(self, mcts=None):
        print("initializing bot...")
        if mcts == None:
            # TODO: load model from file
            self.net = Net()
            # self.net.load(folder='./temp',filename='best.pth.tar')
            self.mcts = None
        else:
            self.mcts = mcts

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        # start = time.time()
        if self.mcts == None:
            self.mcts = MonteCarloSearchTree(asp, self.net)
        # print(time.time() - start)
        # start = time.time()
        state = asp.get_start_state()
        decision = self.mcts.action_map[np.argmax(self.mcts.compute_policy(state, 0))]
        # print("Decision time: ", time.time() - start)
        return decision

    def cleanup(self):
        """
        Input: None
        Output: None

        This function will be called in between
        games during grading. You can use it
        to reset any variables your bot uses during the game
        (for example, you could use this function to reset a
        turns_elapsed counter to zero). If you don't need it,
        feel free to leave it as "pass"
        """
        pass

class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}

        To get started, you can get the current
        state by calling asp.get_start_state()
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"

    def cleanup(self):
        pass


class WallBot:
    """Hugs the wall"""

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"
        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision
