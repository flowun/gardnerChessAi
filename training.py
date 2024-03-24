import os
import pickle
import random
import time
import json
import numpy as np

from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from neural_network import NeuralNetwork
from game import Game
from graph import Graph


"""
How to use:
- choose a MODEL_NAME (or continue training "pretrained")
- define model (in neural_network.py) and set hyperparameters
- run this skript
- model will train
- every few episodes, the model is evaluated against opponents with different strength. The evaluation is automatically 
  visualized in a training graph (if the graph is not open at the end of the evaluation phase)
- you can end this skript whenever you want (all progress is saved at the end of each epoch, this includes both models,
  the replay memory, the training settings with all hyperparameters, the evaluation scores, the training graph and the 
  episode behavior game states)
- when starting this skript again with the same MODEL_NAME as sometime trained before, all the progress will be loaded 
  and the training continues from the beginning of the episode where it was stopped (no problem to stop it for a break)
- if you want to continue the training from the best checkpoint (not the newest model), you can set load_checkpoint=True
"""

MODEL_NAME = "pretrained"
LOAD_CHECKPOINT = False

"""
hyperparameters
(only set when starting a new training, otherwise the saved hyperparameters are loaded)
"""
# episode length
N_STEPS_PER_EPISODE = 1000
# exploration vs exploitation
MIN_EPSILON = 0.1
FINAL_EXPLORATION_FRAME = 500_000
# target model
TAU = 0.0005  # strength of each target model update
STEPS_TO_UPDATE_TARGET_MODEL = 16
# replay memory
MIN_REPLAY_SIZE = 10000
REPLAY_MEMORY_LENGTH = 128_000
USE_PRIORITIZED_REPLAY_MEMORY = False
BIG_BATCH_LENGTH = 10000
PRIORITY_OFFSET = 0.01
A = 0.6
B_MIN, B_MAX = 0.4, 1
# bellman equation
DISCOUNT_FACTOR_BELLMAN = 0.9
LEARNING_RATE_BELLMAN = 1
# training
FITTING_FREQUENCY = 16
LEARNING_RATE_FIT_FUNCTION = 0.0005
LEARNING_RATE_DECAY_STEPS = int(N_STEPS_PER_EPISODE / FITTING_FREQUENCY)
LEARNING_RATE_DECAY_RATE = 0.997
BATCH_SIZE = 128
# evaluation & saving
N_EPOCHS_BETWEEN_EVALUATION = 10
N_EPOCHS_BETWEEN_SAVING = 10
CHECKPOINT_MIN_EPOCHS = 200
# model architecture: build it in neural_network.py in the initialize_model() methode


class Training:

    def __init__(self, model_name, save=True, load_checkpoint=False):
        self.save = save
        self.model_name = model_name
        print("Training model", self.model_name)
        self.define_filepaths()
        self.load_objects(from_checkpoint=load_checkpoint)
        self.main_model = NeuralNetwork(name="main", main_filepath=self.main_filepath, load_from_checkpoint=load_checkpoint, lr_schedule_config=self.lr_schedule_fit_config)
        self.target_model = NeuralNetwork(name="target", main_filepath=self.main_filepath, load_from_checkpoint=load_checkpoint, lr_schedule_config=self.lr_schedule_fit_config)
        self.main_model.model.summary()
        self.training_game = Game(player_1_type='ai_training', player_2_type='ai_training')
        self.training_graph = Graph(model_name=model_name, opponent="all")

    def define_filepaths(self):
        self.main_filepath = "saves/" + self.model_name + "/gardnerChessAi_"
        self.replay_memory_filepath = self.main_filepath + "replay_memory.pkl"
        self.replay_memory_filepath_checkpoint = self.main_filepath + "replay_memory_checkpoint.pkl"
        self.episode_behavior_filepath = self.main_filepath + "episode_behavior.pkl"
        self.episode_behavior_filepath_checkpoint = self.main_filepath + "episode_behavior_checkpoint.pkl"
        self.evaluation_score_filepath = self.main_filepath + "evaluation_scores.pkl"
        self.evaluation_score_filepath_checkpoint = self.main_filepath + "evaluation_scores_checkpoint.pkl"
        self.training_settings_filepath = self.main_filepath + "training_settings.json"
        self.training_settings_filepath_checkpoint = self.main_filepath + "training_settings_checkpoint.json"

    def load_objects(self, from_checkpoint=False):
        """
        try to load all needed pickle objects, initializes them if not available
        """
        print("-----------------------------")
        print("load or create new objects...")
        print()
        if not os.path.exists('saves/' + self.model_name):
            os.makedirs('saves/' + self.model_name)
        self.load_training_settings(from_checkpoint=from_checkpoint)
        self.load_replay_memory(from_checkpoint=from_checkpoint)
        self.load_evaluation_scores(from_checkpoint=from_checkpoint)
        self.load_episode_behavior_game_states(from_checkpoint=from_checkpoint)
        print("loading/creating finished")
        print("-----------------------------")

    def save_training_settings(self, new_starting_episode, checkpoint=False):
        """
        saves the all training settings in a JSON file
        """
        training_settings = {
            "new_starting_episode": new_starting_episode,
            "n_steps_per_episode": self.n_steps_per_episode,
            "min_epsilon": self.min_epsilon,
            "final_exploration_frame": self.final_exploration_frame,
            "steps_to_update_target_model": self.steps_to_update_target_model,
            "tau": self.tau,
            "fitting_frequency": self.fitting_frequency,
            "min_replay_size": self.min_replay_size,
            "use_prioritized_replay_memory": self.use_prioritized_replay_memory,
            "discount_factor_bellman": self.discount_factor_bellman,
            "learning_rate_bellman": self.learning_rate_bellman,
            "batch_size": self.batch_size,
            "lr_schedule_fit_config": self.lr_schedule_fit_config,
            "n_epochs_between_evaluation": self.n_epochs_between_evaluation,
            "n_epochs_between_saving": self.n_epochs_between_saving,
            "checkpoint_min_epochs": self.checkpoint_min_epochs
        }
        path = self.training_settings_filepath_checkpoint if checkpoint else self.training_settings_filepath
        with open(path, "w") as file_handler:
            json.dump(training_settings, file_handler)

    def load_training_settings(self, from_checkpoint=False):
        try:
            path = self.training_settings_filepath_checkpoint if from_checkpoint else self.training_settings_filepath
            with open(path, "r") as file_handler:
                training_settings = json.load(file_handler)
            self.starting_episode = training_settings["new_starting_episode"]
            self.n_steps_per_episode = training_settings["n_steps_per_episode"]
            self.min_epsilon = training_settings["min_epsilon"]
            self.final_exploration_frame = training_settings["final_exploration_frame"]
            self.steps_to_update_target_model = training_settings["steps_to_update_target_model"]
            self.tau = training_settings["tau"]
            self.fitting_frequency = training_settings["fitting_frequency"]
            self.min_replay_size = training_settings["min_replay_size"]
            self.use_prioritized_replay_memory = training_settings["use_prioritized_replay_memory"]
            self.discount_factor_bellman = training_settings["discount_factor_bellman"]
            self.learning_rate_bellman = training_settings["learning_rate_bellman"]
            self.batch_size = training_settings["batch_size"]
            self.lr_schedule_fit_config = training_settings["lr_schedule_fit_config"]
            self.n_epochs_between_evaluation = training_settings["n_epochs_between_evaluation"]
            self.n_epochs_between_saving = training_settings["n_epochs_between_saving"]
            self.checkpoint_min_epochs = training_settings["checkpoint_min_epochs"]
            print("training_settings loaded:", path)
            print(training_settings)
        except FileNotFoundError:
            # training loop
            self.starting_episode = 0
            self.n_steps_per_episode = N_STEPS_PER_EPISODE
            self.min_epsilon = MIN_EPSILON
            self.final_exploration_frame = FINAL_EXPLORATION_FRAME
            self.steps_to_update_target_model = STEPS_TO_UPDATE_TARGET_MODEL
            self.tau = TAU
            # fitting
            self.fitting_frequency = FITTING_FREQUENCY
            self.min_replay_size = MIN_REPLAY_SIZE
            self.use_prioritized_replay_memory = USE_PRIORITIZED_REPLAY_MEMORY
            self.discount_factor_bellman = DISCOUNT_FACTOR_BELLMAN
            self.learning_rate_bellman = LEARNING_RATE_BELLMAN
            self.batch_size = BATCH_SIZE
            self.lr_schedule_fit_config = {"init_lr": LEARNING_RATE_FIT_FUNCTION, "decay_steps": LEARNING_RATE_DECAY_STEPS, "decay_rate": LEARNING_RATE_DECAY_RATE}
            # evaluation and saving
            self.n_epochs_between_evaluation = N_EPOCHS_BETWEEN_EVALUATION
            self.n_epochs_between_saving = N_EPOCHS_BETWEEN_SAVING
            self.checkpoint_min_epochs = CHECKPOINT_MIN_EPOCHS
            print("set training_settings")
        print("-------training-loop---------")
        print("starting_episode:", self.starting_episode)
        print("n_steps_per_episode:", self.n_steps_per_episode)
        print("min_epsilon:", self.min_epsilon)
        print("final_exploration_frame:", self.final_exploration_frame)
        print("steps_to_update_target_model:", self.steps_to_update_target_model)
        print("tau:", self.tau)
        print("-----------fitting-----------")
        print("fitting_frequency:", self.fitting_frequency)
        print("min_replay_size:", self.min_replay_size)
        print("use_prioritized_replay_memory:", self.use_prioritized_replay_memory)
        print("discount_factor_bellman:", self.discount_factor_bellman)
        print("learning_rate_bellman:", self.learning_rate_bellman)
        print("batch_size:", self.batch_size)
        print("lr_schedule_fit_config =", self.lr_schedule_fit_config)
        print("-----------------------------")
        print("-----evaluation-&-saving-----")
        print("n_epochs_between_evaluation:", self.n_epochs_between_evaluation)
        print("n_epochs_between_saving:", self.n_epochs_between_saving)
        print("checkpoint_min_epochs:", self.checkpoint_min_epochs)
        print("-----------------------------")

    def load_replay_memory(self, from_checkpoint=False):
        """
        loads the replay memory
        """
        try:
            path = self.replay_memory_filepath_checkpoint if from_checkpoint else self.replay_memory_filepath
            with open(path, "rb") as file_handler:
                self.replay_memory = pickle.load(file_handler)
            print("replay_memory loaded:", path)
        except:
            # replay memory settings
            replay_memory_length = REPLAY_MEMORY_LENGTH
            big_batch_length = BIG_BATCH_LENGTH
            priority_offset = PRIORITY_OFFSET
            a = A
            b_min, b_max = B_MIN, B_MAX
            if self.use_prioritized_replay_memory:
                self.replay_memory = PrioritizedReplayBuffer(replay_memory_length=replay_memory_length, big_batch_length=big_batch_length,
                                                             priority_offset=priority_offset, a=a, b_min=b_min, b_max=b_max)
            else:
                self.replay_memory = ReplayBuffer(replay_memory_length=replay_memory_length)
            print("new replay_memory created")
        print("---replay-memory-settings----")
        print("replay_memory_length_current:", len(self.replay_memory.replay_memory))
        print("replay_memory_length_max:", self.replay_memory.replay_memory_length)
        if self.use_prioritized_replay_memory:
            print("big_batch_length:", self.replay_memory.big_batch_length)
            print("priority_offset:", self.replay_memory.priority_offset)
            print("a:", self.replay_memory.a)
            print("b_min-->b_max:", str(self.replay_memory.b_min) + "-->" + str(self.replay_memory.b_max))
        print("-----------------------------")

    def save_replay_memory(self, checkpoint=False):
        """
        saves the replay memory
        """
        path = self.replay_memory_filepath_checkpoint if checkpoint else self.replay_memory_filepath
        with open(path, "wb") as file_handler:
            pickle.dump(self.replay_memory, file_handler)

    def load_evaluation_scores(self, from_checkpoint=False):
        """
        loads the evaluation_scores to append the list (or creates a new one of not available)
        can be saved again with the self.save_evaluation_scores() methode
        """
        try:
            path = self.evaluation_score_filepath_checkpoint if from_checkpoint else self.evaluation_score_filepath
            with open(path, "rb") as file_handler:
                self.evaluation_scores = pickle.load(file_handler)
            print("evaluation_scores loaded:", path)
            print(self.evaluation_scores)
        except:
            self.evaluation_scores = [[0.0], [0.0], [0.0], [0.0]]
            print("new evaluation_scores list created")

    def save_evaluation_scores(self, checkpoint=False):
        """
        saves the evaluation_scores in a pickle object
        """
        path = self.evaluation_score_filepath_checkpoint if checkpoint else self.evaluation_score_filepath
        with open(path, "wb") as file_handler:
            pickle.dump(self.evaluation_scores, file_handler)

    def load_episode_behavior_game_states(self, from_checkpoint=False):
        """
        loads the episode_behavior_game_states to append the list (or creates a new one of not available)
        can be saved again with the self.save_episode_behavior_game_states() methode
        """
        try:
            path = self.episode_behavior_filepath_checkpoint if from_checkpoint else self.episode_behavior_filepath
            with open(path, "rb") as file_handler:
                self.episode_behavior_game_states = pickle.load(file_handler)
            print("episode_behavior loaded:", path)
        except:
            self.episode_behavior_game_states = [[], [], [], []]
            print("new episode_behavior list created")

    def save_episode_behavior_game_states(self, checkpoint=False):
        """
        saves episode_behavior_game_states in a pickle object --> can be spectated later to see how the AI behaved in each episode
        """
        path = self.episode_behavior_filepath_checkpoint if checkpoint else self.episode_behavior_filepath
        with open(path, "wb") as file_handler:
            pickle.dump(self.episode_behavior_game_states, file_handler)

    def get_possible_init_moves(self):
        return self.training_game.get_all_legal_moves(self.training_game.get_piece_positions_of_player())

    def get_init_nnet_input(self, white_possible_moves):
        return self.main_model.get_nnet_input(self.training_game.get_all_reachable_boards(white_possible_moves))

    def get_move_with_highest_q_value(self, moves, q_values):
        """
        returns best move (deterministic) and its index
        """
        index = np.argmax(q_values)
        move = moves[index]
        return move, index

    def get_move_result(self, winner):
        """
        returns:
        1. reward: loss-->-1, draw or not done-->0, win-->1
        2. if game is done
        """
        if winner == 0:
            white_reward, black_reward, done = 0, 0, False  # not done
        elif winner == 1:
            white_reward, black_reward, done = 1, -1, True  # white won
        elif winner == 2:
            white_reward, black_reward, done = -1, 1, True  # black won
        else:
            white_reward, black_reward, done = 0, 0, True  # draw
        return white_reward, black_reward, done

    def fit_on_memory(self):
        """
        Trains the main model using the replay memory

        Each experience in the minibatch is a tuple containing the following elements:
            - The state of the game when/before the action was taken (represented as a neural network input)
            - The reward obtained after taking the action
            - The state of the game after the action was taken (also represented as a neural network input)
            - A boolean indicating whether the game ended after the action was taken

        - If the replay memory is big enough, a minibatch is sampled (random or according to replay memory priorities)
        - Q-Values are calculated according to Double DQN algorithm
          (old_Qs with main model and future_Qs of best actions (according to main model) with target model)
        - New Q-Values are calculated with the Bellman equation
        - Main model is trained with new Q-Values
        - Replay memory priorities are updated (if prioritized replay memory is used)
        """
        # return if replay memory is too small
        if len(self.replay_memory.replay_memory) < self.min_replay_size:
            return
        # get minibatch
        if self.use_prioritized_replay_memory:
            mini_batch, replay_memory_indices_of_minibatch, weighting_factors = self.replay_memory.get_prioritized_minibatch(self.batch_size)
        else:
            mini_batch = self.replay_memory.get_random_minibatch(self.batch_size)
        # get best (old) nnet inputs
        best_nnet_inputs = np.array([transition[0] for transition in mini_batch])
        # get old Qs
        old_Qs_main = self.main_model.get_q_values(best_nnet_inputs)
        """
        # get future Qs (not in for loop/all in one single model call due to computation improvement)
        # needs to be concatenated because the model expects the shape (k, 5, 5, 13) and there are 
        # multiple inputs in each batch (one for each possible action - not all the same amount)
        """
        # get new nnet inputs
        new_nnet_inputs = [transition[2] for transition in mini_batch]
        # store lengths of new_nnet_inputs to split concatenated_future_Qs later
        len_new_nnet_inputs = []
        for new_nnet_input in new_nnet_inputs:
            len_new_nnet_inputs.append(len(new_nnet_input))
        # concatenate new_nnet_inputs
        concatenated_new_nnet_inputs = np.concatenate(new_nnet_inputs, axis=0)
        # get future Qs according to main model with a single model() call
        concatenated_future_Qs_main = self.main_model.get_q_values(concatenated_new_nnet_inputs)
        # split concatenated_future_Qs and get best next action indices according to main model
        best_action_indices_main = []
        current_len = 0
        for len_new_nnet_input in len_new_nnet_inputs:
            future_Qs_main = []
            for i in range(len_new_nnet_input):
                future_Qs_main.append(concatenated_future_Qs_main[i + current_len])
            current_len += len_new_nnet_input
            best_action_indices_main.append(np.argmax(future_Qs_main))
        # get the best new nnet inputs according to main model
        best_new_nnet_inputs_main = []
        for i, best_action_index in enumerate(best_action_indices_main):
            best_new_nnet_inputs_main.append(new_nnet_inputs[i][best_action_index])
        # get future Qs according to target model with a single model() call
        future_Qs_target = self.target_model.get_q_values(np.array(best_new_nnet_inputs_main))
        # calculate new Qs with bellman equation and update replay memory priorities
        new_Qs = []
        for i, (best_nnet_input, reward, new_nnet_input, done) in enumerate(mini_batch):
            old_Q = old_Qs_main[i]
            future_Qs = future_Qs_target[i]
            new_Q = (1 - self.learning_rate_bellman) * old_Q + self.learning_rate_bellman * (reward + self.discount_factor_bellman * future_Qs)
            new_Qs.append(new_Q)
            if self.use_prioritized_replay_memory:
                # update replay memory priorities
                error = old_Q - new_Q
                replay_memory_index_of_experience = replay_memory_indices_of_minibatch[i]
                self.replay_memory.replay_memory_priorities[replay_memory_index_of_experience] = abs(error) + self.replay_memory.priority_offset
        if self.use_prioritized_replay_memory:
            self.main_model.fit_model(X_train=best_nnet_inputs, Y_train=np.array(new_Qs), batch_size=self.batch_size, sample_weight=weighting_factors)
        else:
            self.main_model.fit_model(X_train=best_nnet_inputs, Y_train=np.array(new_Qs), batch_size=self.batch_size)

    def train(self):
        print()
        print("--------------------------------------")
        print("--------------------------------------")
        if self.starting_episode == 0:
            print("------------start training------------")
            self.target_model.load_weights_from_other_model(self.main_model)
        else:
            print("----------continue training-----------")
        steps_since_last_target_model_update = 0
        for episode in range(1000000):
            time1 = time.time()
            episode += self.starting_episode
            print("--------------------------------------")
            print("--------------------------------------")
            print("Episode", episode)
            print("--------------------------------------")
            epsilon = 1 - (1 - self.min_epsilon) / self.final_exploration_frame * episode * self.n_steps_per_episode
            if epsilon < self.min_epsilon:
                epsilon = self.min_epsilon
            if self.use_prioritized_replay_memory:
                self.replay_memory.b_current = self.replay_memory.b_min - (self.replay_memory.b_min - self.replay_memory.b_max) / self.final_exploration_frame * episode * self.n_steps_per_episode
                if self.replay_memory.b_current > 1:
                    self.replay_memory.b_current = 1
            n_steps = 0
            while n_steps < self.n_steps_per_episode:
                if random.random() < 0.2:
                    print(str(int(100 * n_steps / self.n_steps_per_episode)) + "% done")
                self.training_game.start_new_game()
                # initialize variables
                game_done = False
                white_nnet_input, black_nnet_input = None, None
                white_best_nnet_input, black_best_nnet_input = None, None
                white_move, black_move = None, None
                white_possible_moves, black_possible_moves = self.get_possible_init_moves(), None
                white_reward, black_reward = None, None
                white_new_nnet_input, black_new_nnet_input = self.get_init_nnet_input(white_possible_moves), None
                white_done, black_done = False, False
                while not game_done:
                    n_steps += 1
                    steps_since_last_target_model_update += 1
                    # board view needs to be reversed for black in order to train with the experience of both players
                    reverse_view = self.training_game.current_player == self.training_game.BLACK_PLAYER
                    # get nnet_input from the board
                    if reverse_view:
                        black_nnet_input = black_new_nnet_input
                    else:
                        white_nnet_input = white_new_nnet_input
                    # choose move
                    if epsilon > random.random():
                        # random exploration
                        if reverse_view:
                            index = random.randint(0, len(black_possible_moves) - 1)
                            black_move = black_possible_moves[index]
                            black_best_nnet_input = black_nnet_input[index]
                        else:
                            index = random.randint(0, len(white_possible_moves) - 1)
                            white_move = white_possible_moves[index]
                            white_best_nnet_input = white_nnet_input[index]
                    else:
                        # exploitation
                        if reverse_view:
                            q_values = self.main_model.get_q_values(black_nnet_input)
                            black_move, index = self.get_move_with_highest_q_value(black_possible_moves, q_values)
                            black_best_nnet_input = black_nnet_input[index]
                        else:
                            q_values = self.main_model.get_q_values(white_nnet_input)
                            white_move, index = self.get_move_with_highest_q_value(white_possible_moves, q_values)
                            white_best_nnet_input = white_nnet_input[index]
                    # do the move
                    winner = self.training_game.do_move(black_move if reverse_view else white_move)
                    white_reward, black_reward, game_done = self.get_move_result(winner)
                    if game_done and reverse_view:
                        black_done = True
                    elif game_done and not reverse_view:
                        white_done = True
                    # get new_nnet_input for replay memory (it's for the other player)
                    if not game_done:
                        if reverse_view:
                            white_possible_moves = self.training_game.get_all_legal_moves(self.training_game.get_piece_positions_of_player())
                            white_new_nnet_input = self.main_model.get_nnet_input(self.training_game.get_all_reachable_boards(white_possible_moves))
                        else:
                            black_possible_moves = self.training_game.get_all_legal_moves(self.training_game.get_piece_positions_of_player())
                            black_new_nnet_input = self.main_model.get_nnet_input(
                                self.training_game.get_all_reachable_boards(black_possible_moves, reverse=True))
                    # append the data from the other player before it will be overwritten on the next move
                    # in the form [nnet_input, reward, new_states as nnet_input, done]
                    if reverse_view:
                        self.replay_memory.append([white_best_nnet_input, white_reward, white_new_nnet_input, game_done])
                    elif black_nnet_input != None:  # only append after there is a black_nnet_input (after whites second turn of a game)
                        self.replay_memory.append([black_best_nnet_input, black_reward, black_new_nnet_input, game_done])
                    if n_steps % self.fitting_frequency == 0:
                        self.fit_on_memory()
                    if steps_since_last_target_model_update >= self.steps_to_update_target_model and episode > self.min_replay_size / self.n_steps_per_episode:
                        steps_since_last_target_model_update = 0
                        # slightly update target model with weights from main model
                        self.target_model.update_target_model_weights(self.main_model, self.tau)
                if white_done:
                    self.replay_memory.append([white_best_nnet_input, white_reward, white_new_nnet_input, game_done])
                elif black_done:
                    self.replay_memory.append([black_best_nnet_input, black_reward, black_new_nnet_input, game_done])
            time2 = time.time()
            print("episode took", int(time2 - time1), "seconds")
            if self.save:
                if episode % self.n_epochs_between_saving == 0 and episode != 0:
                    print("saving model, replay_memory and training_settings ...")
                    self.main_model.save_model()
                    self.target_model.save_model()
                    self.save_replay_memory()
                    self.save_training_settings(new_starting_episode=episode + 1)
                    print("saving completed")
                if episode % self.n_epochs_between_evaluation == 0 and episode != 0:
                    evaluation_game_random = Game(player_1_type='ai_evaluating', player_2_type='random', window=False,
                                                  nnets=[self.main_model, None], ai_decision_temperature=0.1)
                    evaluation_game_50_50 = Game(player_1_type='ai_evaluating', player_2_type='50_minimax_50_random', window=False,
                                                 search_depth=2, nnets=[self.main_model, None], ai_decision_temperature=0.1)
                    evaluation_game_90_10 = Game(player_1_type='ai_evaluating', player_2_type='90_minimax_10_random', window=False,
                                                 search_depth=2, nnets=[self.main_model, None], ai_decision_temperature=0.1)
                    evaluation_game_minimax = Game(player_1_type='ai_evaluating', player_2_type='minimax', window=False,
                                                   search_depth=2, nnets=[self.main_model, None], ai_decision_temperature=0.1)
                    print("---------------")
                    # random opponent
                    print("---------------")
                    print(f"Evaluating against random opponent (Episode {episode})...")
                    n_evaluation_games_random = 50
                    ai_wins, random_wins, draws, board_states_of_a_game, average_game_length = evaluation_game_random.game_loop(n_games=n_evaluation_games_random)
                    if episode % 100 == 0:
                        self.episode_behavior_game_states[1].append(board_states_of_a_game)
                    print("ai_wins:", str(round(100 * ai_wins / n_evaluation_games_random, 1)) + "%")
                    print("random_wins:", str(round(100 * random_wins / n_evaluation_games_random, 1)) + "%")
                    print("draws:", str(round(100 * draws / n_evaluation_games_random, 1)) + "%")
                    evaluation_score = ai_wins + 0.5 * draws
                    self.evaluation_scores[1].append(round(100 * evaluation_score / n_evaluation_games_random, 1))
                    del evaluation_game_random
                    # 50% random and 50% minimax moves
                    print("---------------")
                    print(f"Evaluating against 50% minimax(2) 50% random opponent (Episode {episode})...")
                    n_evaluation_games_50_50 = 50
                    ai_wins, random_wins, draws, board_states_of_a_game, average_game_length = evaluation_game_50_50.game_loop(n_games=n_evaluation_games_50_50)
                    if episode % 100 == 0:
                        self.episode_behavior_game_states[2].append(board_states_of_a_game)
                    print("ai_wins:", str(round(100 * ai_wins / n_evaluation_games_50_50, 1)) + "%")
                    print("50% random, 50% minimax wins:", str(round(100 * random_wins / n_evaluation_games_50_50, 1)) + "%")
                    print("draws:", str(round(100 * draws / n_evaluation_games_50_50, 1)) + "%")
                    evaluation_score = ai_wins + 0.5 * draws
                    self.evaluation_scores[2].append(round(100 * evaluation_score / n_evaluation_games_50_50, 1))
                    del evaluation_game_50_50
                    # 10% random and 90% minimax moves
                    print("---------------")
                    print(f"Evaluating against 90% minimax(2) 10% random opponent (Episode {episode})...")
                    n_evaluation_games_90_10 = 50
                    ai_wins, random_wins, draws, board_states_of_a_game, average_game_length = evaluation_game_90_10.game_loop(n_games=n_evaluation_games_90_10)
                    if episode % 100 == 0:
                        self.episode_behavior_game_states[3].append(board_states_of_a_game)
                    print("ai_wins:", str(round(100 * ai_wins / n_evaluation_games_90_10, 1)) + "%")
                    print("10% random, 90% minimax wins:", str(round(100 * random_wins / n_evaluation_games_90_10, 1)) + "%")
                    print("draws:", str(round(100 * draws / n_evaluation_games_90_10, 1)) + "%")
                    evaluation_score = ai_wins + 0.5 * draws
                    self.evaluation_scores[3].append(round(100 * evaluation_score / n_evaluation_games_90_10, 1))
                    del n_evaluation_games_90_10
                    # minimax opponent
                    print("---------------")
                    print(f"Evaluating against minimax(2) opponent (Episode {episode})...")
                    n_evaluation_games_minimax = 50
                    ai_wins, minimax_wins, draws, board_states_of_a_game, average_game_length = evaluation_game_minimax.game_loop(n_games=50)
                    if episode % 100 == 0:
                        self.episode_behavior_game_states[0].append(board_states_of_a_game)
                    evaluation_score = ai_wins + 0.5 * draws
                    if 100 * evaluation_score / n_evaluation_games_minimax > max(self.evaluation_scores[0]) and episode >= self.checkpoint_min_epochs:
                        n_evaluation_games_minimax += 50
                        ai_wins_add, minimax_add, draws_add, _, _ = evaluation_game_minimax.game_loop(n_games=50)
                        ai_wins, minimax_wins, draws = ai_wins + ai_wins_add, minimax_wins + minimax_add, draws + draws_add
                        print("ai_wins:", str(round(100 * ai_wins / n_evaluation_games_minimax, 1)) + "%")
                        print("minimax_wins:", str(round(100 * minimax_wins / n_evaluation_games_minimax, 1)) + "%")
                        print("draws:", str(round(100 * draws / n_evaluation_games_minimax, 1)) + "%")
                        evaluation_score = ai_wins + 0.5 * draws
                        if 100 * evaluation_score / n_evaluation_games_minimax > max(self.evaluation_scores[0]):
                            self.evaluation_scores[0].append(round(100 * evaluation_score / n_evaluation_games_minimax, 1))
                            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                            print("this model reached a new score peak:", round(100 * evaluation_score / n_evaluation_games_minimax, 1))
                            print("creating checkpoint...")
                            self.main_model.save_model(checkpoint=True)
                            self.target_model.save_model(checkpoint=True)
                            self.save_replay_memory(checkpoint=True)
                            self.save_evaluation_scores(checkpoint=True)
                            self.save_episode_behavior_game_states(checkpoint=True)
                            self.save_training_settings(new_starting_episode=episode + 1, checkpoint=True)
                            self.training_graph.create_graph(self.evaluation_scores)
                            self.training_graph.save(checkpoint=True)
                            print("checkpoint created")
                            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                        else:
                            self.evaluation_scores[0].append(round(100 * evaluation_score / n_evaluation_games_minimax, 1))
                    else:
                        self.evaluation_scores[0].append(round(100 * evaluation_score / n_evaluation_games_minimax, 1))
                        print("ai wins:", str(round(100 * ai_wins / n_evaluation_games_minimax, 1)) + "%")
                        print("minimax wins:", str(round(100 * minimax_wins / n_evaluation_games_minimax, 1)) + "%")
                        print("draws:", str(round(100 * draws / n_evaluation_games_minimax, 1)) + "%")
                    print("average_game_length:", average_game_length)
                    print("---------------")
                    del evaluation_game_minimax
                    print("saving episode_behavior_game_states and evaluation_scores ...")
                    self.save_episode_behavior_game_states()
                    self.save_evaluation_scores()
                    self.training_graph.create_graph(self.evaluation_scores)
                    self.training_graph.save()
                    print("saving completed")
                    print()
                    print("evaluation_scores:")
                    print("random:", self.evaluation_scores[1])
                    print("50% minimax, 50% random:", self.evaluation_scores[2])
                    print("90% minimax, 10% random:", self.evaluation_scores[3])
                    print("minimax:", self.evaluation_scores[0])
                    print()
                    print("evaluation done")
                    print("---------------")
                    print("---------------")
                    print("continue training...")

if __name__ == '__main__':
    trainer = Training(MODEL_NAME, save=True, load_checkpoint=LOAD_CHECKPOINT)
    trainer.train()
