import math
import numpy as np
import logging
from agent import Net
from preprocess import convert_board, pad_board
from hyperparameters import MCTS_PARAMETERS

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
        self.Vs = {}  # stores game.getValidMoves for board s

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
            valid_moves = self.asp.get_safe_actions(state.board, state.player_locs[state.ptm])
            valid_vector = np.array([1 if action in valid_moves else 0 for action in self.action_map])
            self.Ps[state] *= valid_vector # Making sure to remove impossible moves
            
            # TODO: Replace with normalize function if no errors
            state_sum = np.sum(self.Ps[state])
            if state_sum > 0:
                self.Ps[state] /= state_sum
            else:
                # if all valid moves were masked make all valid moves equally probable
                # a suggested workaround, hitting this line suggests something is wrong with code
                log.error("All valid moves were masked, doing a workaround.")
                log.info("loc: %s", state.player_locs[state.ptm])
                log.info("available moves: %s" % valid_moves)
                self.Ps[state] += valid_vector
                self.Ps[state] /= np.sum(self.Ps[state])

            self.Vs[state] = valid_vector
            self.Ns[state] = 0
            return -v
        
        valid_vector = self.Vs[state]
        curr = -float('inf')
        best_action = -1

        for action_ind in range(self.num_actions):
            if valid_vector[action_ind]:
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
        for i in range(self.num_sim):
            self.search(state)
        
        action_counts = [self.Nsa[(state, action)] if (state, action) in self.Nsa else 0 for action in range(self.num_actions)]

        if temperature == 0:
            best_action_index = np.random.choice(np.array(np.argwhere(action_counts == np.max(action_counts))).flatten())
            policy = np.zeros(len(action_counts))
            policy[best_action_index] = 1
            return policy

        policy = np.array([count ** 1/float(temperature) for count in action_counts])
        return normalize(policy)


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