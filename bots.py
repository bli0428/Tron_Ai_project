#!/usr/bin/python

import numpy as np
from numpy.lib.polynomial import polysub
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
        voronoi = 0

        # ptm doesn't matter for tarjan, just need a starting point
        tarjan = Tarjan(asp, state.board)
        tarjan.find_articulation_points(locs[ptm])
        ap = tarjan.get_articulation_points()

        battlefield = set()

        ptm_dists = self.dijkstra(asp, state, locs[ptm])
        opp_dists = self.dijkstra(asp, state, locs[opp])
        for r in range(ptm_dists.shape[0]):
            for c in range(ptm_dists.shape[1]):
                if ptm_dists[r,c] != 0 and ptm_dists[r,c] == opp_dists[r,c]:
                    battlefield.add((r,c))
                if ptm_dists[r,c] < opp_dists[r,c]:
                    voronoi +=1
                elif ptm_dists[r,c] > opp_dists[r,c]:
                    voronoi -= 1

        visited = np.zeros(np.shape(state.board))
        num_spaces, is_battlefield, adj_ap = self.count_spaces(asp, state, locs[ptm], visited, ap, battlefield)
        if is_battlefield:
            return voronoi
        else:
            ptm_counts = []
            for adj in adj_ap:
                chain_adj = adj
                while True:
                    adj_num_spaces, adj_is_battlefield, adj_next_ap = self.count_spaces(asp, state, chain_adj, visited, ap, battlefield)
                    if adj_is_battlefield == True:
                        ptm_counts.append(0)
                        break
                    elif adj_num_spaces == 0 and len(list(adj_next_ap)) == 1 and visited[list(adj_next_ap)[0]] == False:
                        chain_adj = list(adj_next_ap)[0]
                    else:
                        ptm_counts.append(adj_num_spaces)
                        break
            ptm_spaces = num_spaces + max(ptm_counts)

            visited = np.zeros(np.shape(state.board))
            num_spaces, is_battlefield, adj_ap = self.count_spaces(asp, state, locs[opp], visited, ap, battlefield)
            opp_counts = []
            for adj in adj_ap:
                chain_adj = adj
                while True:
                    adj_num_spaces, adj_is_battlefield, adj_next_ap = self.count_spaces(asp, state, chain_adj, visited, ap, battlefield)
                    if adj_is_battlefield == True:
                        opp_counts.append(0)
                        break
                    elif adj_num_spaces == 0 and len(list(adj_next_ap)) == 1 and visited[list(adj_next_ap)[0]] == False:
                        chain_adj = list(adj_next_ap)[0]
                    else:
                        opp_counts.append(adj_num_spaces)
                        break
            opp_spaces = num_spaces + max(opp_counts)

            return ptm_spaces - opp_spaces
            

                    
    
    def count_spaces(self, asp, state, pos, visited, articulation_points, battlefield_points):
        board = state.board
        is_battlefield = False
        count = 0
        frontier = Queue()
        frontier.put(pos)
        visited[pos] = True
        next_articulation_points = set()
        while not frontier.empty():
            loc = frontier.get()
            if loc in battlefield_points:
                is_battlefield = True
            if loc not in articulation_points:
                count += 1
            actions = asp.get_safe_actions(board, loc)
            for direction in actions:
                next_loc = asp.move(loc, direction)
                if next_loc in articulation_points:
                    next_articulation_points.add(next_loc)
                if not visited[next_loc] and next_loc not in articulation_points:
                    frontier.put(next_loc)
                    visited[next_loc] = True
        return count, is_battlefield, next_articulation_points

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
class Tarjan:

    "Implements Tarjan's algorithm for finding articulation points"
    def __init__(self, asp, board):
        """
        Input:
            board- a list of lists of characters representing cells
                ('#' for wall, ' ' for space, etc.)
            player_locs- a list of tuples (representing the players' locations)
            ptm- the player whose move it is. player_locs and ptm are
                indexed the same way, so player_locs[ptm] would
                give the location of the player whose move it is.
            player_powerups- a map from player to a map of what powerups they have
                {player : {PowerupType : powerup value}}
        """
        self.asp = asp
        self.board = board
        board_shape = np.shape(board)
        self.articulation_points = set()
        self.discovery_time = np.zeros(board_shape)
        self.low = np.zeros(board_shape)
        self.visited = np.zeros(board_shape)
        self.parents = np.empty(board_shape, dtype=object)
        self.time = 0
    
    def find_articulation_points(self, loc):
        self.visited[loc] = True
        self.time += 1
        self.low[loc] = self.time
        self.discovery_time[loc] = self.time
        child = 0
        for direction in self.asp.get_safe_actions(self.board, loc):
            next_loc = self.asp.move(loc, direction)
            if not self.visited[next_loc]:
                child += 1
                self.parents[next_loc] = loc
                self.find_articulation_points(next_loc)
                self.low[loc] = min(self.low[loc], self.low[next_loc])
                if self.parents[loc] == None and child > 1:
                    self.articulation_points.add(loc)
                if self.parents[loc] != None and self.low[next_loc] >= self.discovery_time[loc]:
                    self.articulation_points.add(loc)
            elif next_loc != self.parents[loc]:
                self.low[loc] = min(self.low[loc], self.discovery_time[next_loc])

    def get_articulation_points(self):
        return self.articulation_points

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