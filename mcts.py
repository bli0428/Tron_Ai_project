import math
import numpy as np
import logging

from torch.functional import norm
from agent import Net
from preprocess import convert_board, pad_board
from hyperparameters import MCTS_PARAMETERS
from tronproblem import TronProblem

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MonteCarloSearchTree():
    def __init__(self, asp, net):

        self.asp = asp
        self.net = net
        self.num_actions = 4
        self.c_puct = MCTS_PARAMETERS["c_puct"]
        self.num_sim = MCTS_PARAMETERS["num_sim"]
        self.epsilon = MCTS_PARAMETERS["epsilon"]
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s

        self.action_map = ['U','D','L','R']

    def search(self, state):
        if state not in self.Es:
            if self.asp.is_terminal_state(state):
                v_s = self.asp.evaluate_state(state)[state.ptm]
                if v_s:
                    self.Es[state] = 1
                else:
                    self.Es[state] = -1
            else:
                self.Es[state] = 0
        if self.Es[state] != 0:
            return -self.Es[state]
                
        
        if state not in self.Ps:
            converted_board = convert_board(state.board, state.ptm)
            padded_board = pad_board(converted_board)

            self.Ps[state], v = self.net.predict(padded_board)
            
            self.Ps[state] = normalize(self.Ps[state])
            self.Ns[state] = 0
            return -v
        
        curr = -float('inf')
        best_action = -1

        for action_ind in range(self.num_actions):
            ucb = self.ucb(state, action_ind)
            if ucb > curr:
                curr = ucb
                best_action = action_ind

        next_state = self.asp.transition(state, self.action_map[best_action])

        v = self.search(next_state)
        if (state, best_action) in self.Qsa:
            self.Qsa[(state, best_action)] = (self.Nsa[(state, best_action)] * self.Qsa[(state, best_action)] + v) / (self.Nsa[(state, best_action)] + 1)
            self.Nsa[(state, best_action)] += 1
        else:
            self.Qsa[(state, best_action)] = v
            self.Nsa[(state, best_action)] = 1

        self.Ns[state] += 1
        return -v
    
    def compute_policy(self, state, temperature):
        # Computes the policy purely with the neural net, with no MCTS
        converted_board = convert_board(state.board, state.ptm)
        padded_board = pad_board(converted_board)
        pred_policy, v = self.net.predict(padded_board)

        if temperature == 0:
            best_action_index = np.random.choice(np.array(np.argwhere(pred_policy == np.max(pred_policy))).flatten())
            policy = np.zeros(len(pred_policy))
            policy[best_action_index] = 1
            return policy

        return normalize(pred_policy)


    def ucb(self, state, action):
        # Calculates the upper confidence bound, note: action is index with a corresponding index number, not the action string itself
        if (state, action) in self.Qsa:
            return self.Qsa[state, action] + self.c_puct * self.Ps[state][action] * math.sqrt(self.Ns[state]) / (
                1 + self.Nsa[state, action])
        else:
            return self.c_puct * self.Ps[state][action] * math.sqrt(self.Ns[state] + self.epsilon) 

def normalize(arr):
    sum = np.sum(arr)
    if sum == 0:
        sum = 1
    return arr / float(sum)