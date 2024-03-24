import os
import pickle
import matplotlib.pyplot as plt


class Graph:

    def __init__(self, model_name, opponent="all"):
        self.model_name = model_name
        self.opponent = opponent
        self.filepath_evaluation_scores = "saves/" + model_name + "/gardnerChessAi_evaluation_scores.pkl"
        self.filepath_graph = "saves/" + model_name + "/gardnerChessAi_training_graph_" + model_name + ".pdf"
        self.filepath_graph_checkpoint = "saves/" + model_name + "/gardnerChessAi_training_graph_" + model_name + "_checkpoint.pdf"

    def load_evaluation_scores(self):
        with open(self.filepath_evaluation_scores, "rb") as file_handler:
            return pickle.load(file_handler)

    def create_graph(self, evaluation_scores=None):
        if evaluation_scores is None:
            evaluation_scores = self.load_evaluation_scores()
        plt.style.use('seaborn-v0_8-darkgrid')  # if this doesn't work on windows, use 'seaborn-darkgrid'
        plt.xlabel("episode (1000 steps per episode)")
        plt.ylabel("score in % (score = wins + 0.5 * draws)")
        plt.title("training evaluation of model '" + self.model_name + "' (temperature = 0.1)", fontsize=13)
        epochs = [10 * i for i in range(len(evaluation_scores[0]))]
        scores_random = evaluation_scores[1]
        scores_50_50 = evaluation_scores[2]
        scores_90_10 = evaluation_scores[3]
        scores_minimax = evaluation_scores[0]

        if self.opponent == "random" or self.opponent == "all":
            plt.plot(epochs, scores_random, color='tab:green')
        if self.opponent == "50_minimax_50_random" or self.opponent == "all":
            plt.plot(epochs, scores_50_50, color='tab:orange')
        if self.opponent == "90_minimax_10_random" or self.opponent == "all":
            plt.plot(epochs, scores_90_10, color='tab:blue')
        if self.opponent == "minimax" or self.opponent == "all":
            plt.plot(epochs, scores_minimax, color='tab:red')
        if self.opponent == "all":
            plt.legend(['random', '50_minimax_50_random', '90_minimax_10_random', 'minimax(2)'], title="opponent")

    def show(self):
        plt.show()

    def save(self, checkpoint=False):
        path = self.filepath_graph_checkpoint if checkpoint else self.filepath_graph
        try:
            plt.savefig(path)
        except:
            pass
        plt.clf()

    def open_saved_figure(self, from_checkpoint=False):
        path = self.filepath_graph_checkpoint if from_checkpoint else self.filepath_graph
        if os.path.exists(path):
            os.startfile(path)
        else:
            print("file does not exist")
