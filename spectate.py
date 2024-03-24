import pickle

from game import Game

"""
Description
-----------
Every hundred epochs of the training, the evaluation games were saved. Here you can watch them to see the progress 
in the behavior of the AI over time. You can choose between the following evaluation opponents:
"minimax", "random", "50_minimax_50_random", "90_minimax_10_random"
Furthermore, you can choose if you want to watch the games where the AI played as white or black: 
"white", "black"
After running the script, you can explore the moves with the <- and -> arrow keys. To skip to the next game which is
100 epochs later, click "next game".
-----------
"""

# specify model name e.g. "pretrained"
MODEL_NAME = "pretrained"

# specify evaluation opponent e.g. "minimax"
OPPONENT = "minimax"

# specify color of AI e.g. "white"
AI_COLOR = "white"


# ------------------------ no need to change anything below this line ------------------------
episode_behavior_filepath = "saves/" + MODEL_NAME + "/gardnerChessAi_episode_behavior.pkl"
try:
    with open(episode_behavior_filepath, "rb") as file_handler:
        episode_behavior_game_states = pickle.load(file_handler)
except FileNotFoundError:
    print("The file", episode_behavior_filepath, "does not exist.")
    exit()
game = Game(window=True)
game.window.spectate(episode_behavior_game_states, ai_color=AI_COLOR, opponent=OPPONENT)
