#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math
from queue import Queue, PriorityQueue
from typing import Any
from dataclasses import dataclass, field
# Throughout this file, ASP means adversarial search problem.

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)
class StudentBot:
    """ Write your student bot here"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        return self.alpha_beta_cutoff(asp, 3, self.heuristic_func)
    
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

    def alpha_beta_cutoff(self, asp, cutoff_ply, heuristic_func):
        """
        This function should:
        - search through the asp using alpha-beta pruning
        - cut off the search after cutoff_ply moves have been made.

        Input:
            asp - an AdversarialSearchProblem
            cutoff_ply - an Integer that determines when to cutoff the search and
                use heuristic_func. For example, when cutoff_ply = 1, use
                heuristic_func to evaluate states that result from your first move.
                When cutoff_ply = 2, use heuristic_func to evaluate states that
                result from your opponent's first move. When cutoff_ply = 3 use
                heuristic_func to evaluate the states that result from your second
                move. You may assume that cutoff_ply > 0.
            heuristic_func - a function that takes in a GameState and outputs a
                real number indicating how good that state is for the player who is
                using alpha_beta_cutoff to choose their action. You do not need to
                implement this function, as it should be provided by whomever is
                calling alpha_beta_cutoff, however you are welcome to write
                evaluation functions to test your implemention. The heuristic_func
                we provide does not handle terminal states, so evaluate terminal
                states the same way you evaluated them in the previous algorithms.
        Output:
            an action(an element of asp.get_available_actions(asp.get_start_state()))
        """
        start_state = asp.get_start_state()
        maximizing_player = start_state.ptm
        if maximizing_player == 0:
            maximizing_player = -1
        def max_value(state: GameState, alpha: float, beta: float, c_ply: int) -> float:
            if asp.is_terminal_state(state):
                result = asp.evaluate_state(state)
                # Since we can assume the games are alternating, I can just make the numbers negative
                return 1000*maximizing_player * (result[1] - result[0])
            else:
                v = -math.inf
                for a in list(asp.get_safe_actions(state.board, state.player_locs[state.ptm])):
                    if c_ply <= 0:
                        v = max(v, heuristic_func(asp, state))
                    else:
                        v = max(v, min_value(asp.transition(state, a), alpha, beta, c_ply-1))
                        if v >= beta:
                            return v
                        alpha = max(alpha, v)
                return v

        def min_value(state: GameState, alpha: float, beta: float, c_ply: int) -> float:
            if asp.is_terminal_state(state):
                result = asp.evaluate_state(state)
                return 1000*maximizing_player * (result[1] - result[0])
            else:
                v = math.inf
                for a in list(asp.get_safe_actions(state.board, state.player_locs[state.ptm])):
                    if c_ply <= 0:
                        v = min(v, heuristic_func(asp, state))
                    else: 
                        v = min(v, max_value(asp.transition(state, a), alpha, beta, c_ply-1))
                        if v <= alpha:
                            return v
                        beta = min(beta, v)
                return v

        best_action = None
        curr_v = -math.inf
        alpha = -math.inf
        beta = math.inf
        for action in list(asp.get_safe_actions(start_state.board, start_state.player_locs[start_state.ptm])):
            next_state = asp.transition(start_state, action)
            if cutoff_ply == 1:
                val = heuristic_func(asp, next_state)
                if (val > curr_v):
                    curr_v = val
                    best_action = action
            else:
                val = min_value(next_state, alpha, beta, cutoff_ply-1)
                if (val > curr_v):
                    curr_v = val
                    best_action = action
                alpha = max(alpha, curr_v)
        return best_action

    def heuristic_func(self, asp, state):
        ptm = state.ptm
        opp = (state.ptm+1)%2
        locs = state.player_locs
        total = 0
        ptm_dists = self.dijkstra(asp, state, locs[ptm])
        opp_dists = self.dijkstra(asp, state, locs[opp])
        for r in range(ptm_dists.shape[0]):
            for c in range(ptm_dists.shape[1]):
                # if ptm_dists[r,c] == 0:
                #     continue
                if ptm_dists[r,c] < opp_dists[r,c]:
                    total +=1
                elif ptm_dists[r,c] > opp_dists[r,c]:
                    total -= 1
                # elif ptm_dists[r,c] == opp_dists[r,c]:
                #     total += 0.5
        return total


    def dijkstra(self, asp, state, pos):
        board = state.board
        board_shape = np.array(board).shape
        # a dict of the cheapest paths from start to a given node
        g = {}
        g[pos] = 0
        result = np.zeros(board_shape)
        
        frontier = PriorityQueue()
        frontier.put(PrioritizedItem(0, pos))
        while not frontier.empty():
            loc = frontier.get().item 
            actions = asp.get_safe_actions(board, loc)
            for direction in actions:
                next_loc = asp.move(loc, direction)
                curr_g = g[loc] + 1
                if next_loc not in g or curr_g < g[next_loc]:
                    g[next_loc] = curr_g
                    priority = g[next_loc]
                    result[next_loc] = curr_g
                    frontier.put(PrioritizedItem(priority, next_loc))
        return result
        
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