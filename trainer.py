
from agent import Net
from mcts import MonteCarloSearchTree
import logging
import numpy as np
from preprocess import get_converted_boards
from bots import StudentBot
from tronproblem import TronProblem
from random import shuffle
import copy
import os
import sys
from pickle import Pickler, Unpickler
from hyperparameters import TRAINER_PARAMETERS

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Trainer:
    """
    Trainer class for handling the model learning
    Note to self: Assumes that player is always player 0, and therefore all data before training should be adjusted
    such that it is from the perspective of player 0 (as in ptm is player 0)
    """
    def __init__(self, maps, net):
        self.maps = maps
        self.net = net 
        self.prev_net = Net()
        self.train_history = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skip_first_self_play = False  # can be overriden in load_train_examples()
        self.num_iterations = TRAINER_PARAMETERS["num_iterations"]
        self.num_episodes = TRAINER_PARAMETERS["num_episodes"]
        self.num_games = TRAINER_PARAMETERS["num_games"] # how many games to run to compare models (for each map)
        self.update_threshold = TRAINER_PARAMETERS["update_threshold"]
        self.train_history_size = TRAINER_PARAMETERS["train_history_size"]
        self.temperature_threshold = TRAINER_PARAMETERS["temp_threshold"]
        self.skip_first_self_play = False

        self.checkpoint_dir = './temp/'
                                 
    def execute_episode(self, asp, mcts):
        examples = []
        curr_timestep = 0

        state = asp.get_start_state()
        while not (asp.is_terminal_state(state)):
            curr_timestep += 1
            temp = int(curr_timestep < self.temperature_threshold)

            pi = mcts.compute_policy(state, temp) 
            examples.append([state.board, state.ptm, pi])
            
            choice = np.random.choice(len(pi), p=pi)
            result_state = asp.transition(state, mcts.action_map[choice])

            asp.set_start_state(result_state)
            state = result_state
        outcome = asp.evaluate_state(state)
        winner = outcome.index(1)
        output = []
        for e in examples:
            output.extend(get_converted_boards(e[0],e[2],e[1], winner))
        print(output[0])
        return output
    def learn(self):
        for i in range(self.num_iterations):
            log.info(f'Starting Iteration {i+1} ...')
            if not self.skip_first_self_play or i > 0:
                iteration_train_examples = []
                for ep in range(self.num_episodes):
                    # log.info(f'Running Episode {ep + 1}')
                    map_path = np.random.choice(self.maps)
                    game = TronProblem(f'./maps/{map_path}.txt', 0)
                    mcts = MonteCarloSearchTree(game, self.net)  # reset search tree
                    episode = self.execute_episode(game, mcts)
                    iteration_train_examples.append(episode)
                self.train_history.append(iteration_train_examples)

            if len(self.train_history) > self.train_history_size:
                log.warning(
                    f"Removing the oldest entry in train history. len(train_history) = {len(self.train_history)}")
                self.train_history.pop(0)
            
            self.save_train_examples(i)

            train_examples = []
            for e in self.train_history:
                train_examples.extend(e)
            shuffle(train_examples)

            self.net.save(folder=self.checkpoint_dir, filename='temp.pth.tar')
            self.prev_net.load(folder=self.checkpoint_dir, filename='temp.pth.tar')

            self.net.train_model(train_examples)

            log.info('Playing previous version...')
            n_wins, win_ratios = self.run_games()

            log.info('Win ratios: %s' % win_ratios)
            if n_wins < len(self.maps):
                log.info('Rejecting new model')
                self.net.load(folder=self.checkpoint_dir, filename='temp.pth.tar')
            else:
                log.info('Accepting new model')
                self.net.save(folder=self.checkpoint_dir, filename=self.get_checkpoint_file(i))
                self.net.save(folder=self.checkpoint_dir, filename='best.pth.tar')

    def run_games(self):
        n_map_wins = 0
        map_win_ratios = {}
        for map_path in self.maps:
            p_bot_wins = 0
            n_bot_wins = 0
            for _ in range(self.num_games):
                asp = TronProblem(f'./maps/{map_path}.txt', 0)
                p_mcts = MonteCarloSearchTree(asp, self.prev_net)
                n_mcts = MonteCarloSearchTree(asp, self.net)
                p_bot = StudentBot(p_mcts)
                n_bot = StudentBot(n_mcts)
                g = self.run_game(asp, [p_bot, n_bot])
                p_bot_wins += g[0]
                n_bot_wins += g[1]
            win_ratio = float(n_bot_wins) / (p_bot_wins + n_bot_wins)
            if win_ratio >= self.update_threshold:
                n_map_wins += 1
            map_win_ratios[map_path] = win_ratio
        return n_map_wins, map_win_ratios

    def run_game(self, asp, bots):
        """
        Inputs:
            - asp: an adversarial search problem
            - bots: a list in which the i'th element is the bot for player i
            - visualizer (optional): a void function that takes in a game state
            and does something to visualize it. If no visualizer argument is
            passed, run_game will not visualize games.

        A copy of the run_game function from gamerunner, with the timer functionality removed for simplicity's sake
        """
        state = asp.get_start_state()

        while not (asp.is_terminal_state(state)):
            exposed = copy.deepcopy(asp)
            decision = bots[state.ptm].decide(exposed)

            available_actions = asp.get_available_actions(state)
            if not decision in available_actions:
                decision = list(available_actions)[0]

            result_state = asp.transition(state, decision)
            asp.set_start_state(result_state)
            state = result_state
        return asp.evaluate_state(state)


    def get_checkpoint_file(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def save_train_examples(self, iteration):
        folder = './temp/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_history)
        f.closed

    def load_train_examples(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        model_file = os.path.join(folder, filename)
        examples_file = model_file + ".examples"
        if not os.path.isfile(examples_file):
            log.warning(f'File "{examples_file}" with train examples not found! Exiting...')
            sys.exit()
        else:
            log.info("File with train examples found. Loading it...")
            with open(examples_file, "rb") as f:
                self.train_history = Unpickler(f).load()
            log.info('Loading done!')
            
            self.skip_first_self_play = True
