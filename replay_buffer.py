import random
from collections import deque
import numpy as np


class PrioritizedReplayBuffer:

    def __init__(self, replay_memory_length, big_batch_length, priority_offset, a, b_min, b_max):
        self.replay_memory_length = replay_memory_length
        self.big_batch_length = big_batch_length
        self.replay_memory = deque(maxlen=self.replay_memory_length)
        self.replay_memory_priorities = deque(maxlen=self.replay_memory_length)
        self.priority_offset = priority_offset
        self.a = a
        self.b_min = b_min
        self.b_max = b_max
        self.b_current = b_min
        """
        priority_offset: small offset value that prevents a priority of 0, because the experience might be useful later on
        a: priority scaling constant between 0 and 1; a=1: full priority sampling, a=0: random sampling
        b: bias correction exponent between 0 and 1; b=1: full bias correction, b=0: no bias correction
           bias correction gets more important over time --> start low for quick learning at the start and increase it to 1 over time
        """

    def append(self, experience):
        """
        takes experience list in the form [observation, reward, new observation, game done]
        appends the replay memory and the memory priorities
        """
        self.replay_memory.append(experience)
        if len(self.replay_memory_priorities) != 0:
            self.replay_memory_priorities.append(np.max(self.replay_memory_priorities))
        else:
            self.replay_memory_priorities.append(1)

    def get_prioritized_minibatch(self, batch_size):
        """
        returns:
        - a minibatch according to the priority distribution of a random sampled part of the replay memory with the
          length self.big_batch_length --> prioritized, but computational more efficient
        - the corresponding indices of the replay memory
        - the weighting factors for the training to reduce bias
        """
        minibatch = []
        replay_memory_indices_of_minibatch = []
        weighting_factors = []
        big_batch_indices = random.sample(range(len(self.replay_memory)),
                                          self.big_batch_length if self.big_batch_length < len(
                                              self.replay_memory) else len(self.replay_memory))
        probabilities = self.get_probability_distribution(big_batch_indices)
        probability_sum_to_index = 0
        random_numbers = np.random.rand(batch_size)
        for big_batch_index in range(len(big_batch_indices)):
            probability = probabilities[big_batch_index]
            for random_number_index in range(len(random_numbers)):
                if random_numbers[random_number_index] < probability + probability_sum_to_index:
                    minibatch.append(self.replay_memory[big_batch_indices[big_batch_index]])
                    replay_memory_indices_of_minibatch.append(big_batch_indices[big_batch_index])
                    weighting_factors.append((1 / (self.big_batch_length * probability)) ** self.b_current)
                    random_numbers[random_number_index] = 20  # set value above 1 to not be reselected
            probability_sum_to_index += probability
        weighting_factors = np.array(weighting_factors)
        weighting_factors /= np.max(weighting_factors)
        return minibatch, replay_memory_indices_of_minibatch, weighting_factors

    def get_probability_distribution(self, big_batch_indices):
        """
        returns an array of the probabilities of choosing the memory for the minibatch
        """
        priorities = np.array([self.replay_memory_priorities[i] for i in big_batch_indices])
        priority_sum = np.sum(priorities ** self.a)
        probabilities = np.array(priorities) ** self.a / priority_sum
        return probabilities


class ReplayBuffer:

    def __init__(self, replay_memory_length):
        self.replay_memory_length = replay_memory_length
        self.replay_memory = deque(maxlen=self.replay_memory_length)

    def append(self, experience):
        """
        takes experience list in the form [observation, reward, new observation, game done]
        appends the replay memory
        """
        self.replay_memory.append(experience)

    def get_random_minibatch(self, batch_size):
        """
        returns a random minibatch
        """
        return random.sample(self.replay_memory, batch_size)
