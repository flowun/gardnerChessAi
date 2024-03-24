import os
import random

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
from keras import Sequential
from keras.layers import Flatten, Dense, Conv2D
from keras.losses import Huber
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.initializers import HeUniform, Zeros


class NeuralNetwork:

    def __init__(self, main_filepath, name='main', load_from_checkpoint=False, lr_schedule_config=None):
        self.name = name
        self.main_filepath = main_filepath
        self.load_from_checkpoint = load_from_checkpoint
        self.model_filepath = main_filepath + "model_" + name
        self.model_filepath_checkpoint = main_filepath + "model_" + name + "_checkpoint"
        self.lr_schedule_config = lr_schedule_config
        self.model = self.get_model()

    def get_model(self):
        """
        tries to load a model
        creates a new model if model couldn't be loaded
        returns model
        """
        try:
            # load model
            path = self.model_filepath_checkpoint if self.load_from_checkpoint else self.model_filepath
            model = tf.keras.models.load_model(filepath=path)
            print(self.name + " model loaded:", path)
        except:
            # create new model
            model = self.initialize_model()
            print("new", self.name, " model created")
        return model

    def initialize_model(self):
        """
        returns the compiled model
        """
        """
        # for games with a bigger board, one can also use a model with Conv2D layers
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=HeUniform(),
                         input_shape=(5, 5, 13)))
        model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer=HeUniform()))
        model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', activation='relu', kernel_initializer=HeUniform()))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer=HeUniform()))  # evt. 512
        model.add(Dense(1, activation='linear', kernel_initializer=Zeros()))
        lr_schedule = ExponentialDecay(initial_learning_rate=self.lr_schedule_config['init_lr'],
                                       decay_steps=self.lr_schedule_config['decay_steps'],
                                       decay_rate=self.lr_schedule_config['decay_rate'])
        model.compile(optimizer=RMSprop(learning_rate=lr_schedule), loss=Huber())
        """
        # this was used for the pretrained model
        model = Sequential()
        model.add(Flatten(input_shape=(5, 5, 13)))
        model.add(Dense(128, activation='relu', kernel_initializer=HeUniform()))
        model.add(Dense(96, activation='relu', kernel_initializer=HeUniform()))
        model.add(Dense(64, activation='relu', kernel_initializer=HeUniform()))
        model.add(Dense(32, activation='relu', kernel_initializer=HeUniform()))
        model.add(Dense(1, activation='linear', kernel_initializer=Zeros()))
        lr_schedule = ExponentialDecay(initial_learning_rate=self.lr_schedule_config['init_lr'],
                                       decay_steps=self.lr_schedule_config['decay_steps'],
                                       decay_rate=self.lr_schedule_config['decay_rate'])
        model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=Huber())
        return model


    def load_weights_from_other_model(self, other_model):
        """
        loads the weights from the other model and overwrites its own weights with it
        """
        self.model.set_weights(other_model.model.get_weights())

    def update_target_model_weights(self, other_model, tau):
        """
        updates the target model weights with the weights of the other model
        new weights = tau * other_model_weights + (1 - tau) * own_weights
        """
        other_model_weights = other_model.model.get_weights()
        own_weights = self.model.get_weights()
        for i in range(len(own_weights)):
            own_weights[i] = tau * other_model_weights[i] + (1 - tau) * own_weights[i]
        self.model.set_weights(own_weights)

    def save_model(self, checkpoint=False):
        """
        saves the model
        """
        path = self.model_filepath_checkpoint if checkpoint else self.model_filepath
        self.model.save(filepath=path)

    def get_nnet_input(self, boards):
        """
        converts boards into nnet input
        """
        return tf.one_hot(indices=boards, depth=13)

    def get_q_values(self, nnet_input):
        """
        returns Q values for the given states represented by the nnet input
        """
        q_values = self.model(nnet_input).numpy().reshape(-1)
        return q_values

    def get_probabilistic_move(self, moves, q_values, old_board_nnet_input, temperature):
        """
        returns a move that the AI would pick using the softmax distribution and its index
        temperature [0-1]:
        small --> confident and deterministic decisions (plays with "best" strategy)
        high --> unconfident and more random decisions
        """
        move_probabilities = []
        old_board_value = self.get_q_values(old_board_nnet_input)
        move_value_differences = q_values - old_board_value
        sum = np.sum(np.exp(move_value_differences / temperature))
        for move_value_difference in move_value_differences:
            move_probabilities.append(np.exp(move_value_difference / temperature) / sum)
        random_number = random.random()
        for i in range(len(move_probabilities)):
            if random_number <= move_probabilities[i]:
                return moves[i], i
            else:
                random_number -= move_probabilities[i]
        # if no move selected due to overflow, return move with the highest probability
        move_index = np.argmax(move_probabilities)
        return moves[move_index], move_index

    def fit_model(self, X_train, Y_train, batch_size, epochs=1, verbose=False, sample_weight=None):
        """
        fitting the model
        """
        self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, sample_weight=sample_weight)
