from game import Game
from neural_network import NeuralNetwork

"""
-----------
Description
-----------
You can use this skript to play games against the AI or let the AI play against some predefined opponents. After every
game, the color of the players switches, so that both players have the same amount of games as white and black.
You can choose between the following player types: "real", "ai", "random", "minimax", "ai+minimax2", '50_minimax_50_random'.

When choosing "ai" or "ai+minimax2", you can specify the model name (same name it was trained on) and if the best 
checkpoint should be loaded (otherwise the newest model is loaded, which might be worse than the best checkpoint).
Additionally for "ai", you can specify the AI decision temperature as a float (0 is deterministic, 1 is more random).

When choosing "minimax", you can specify the search depth as an integer.

For visualization, you can choose use_window=True to see the game in a window and specify the time delay between moves.
If you only want to see the results of a lot of games which the AI played against an opponent, you can set 
use_window=False and specify the number of games the players should play. The results will be printed at the end.
-----------
Examples
-----------
E.g. 1: playing against the best pretrained AI with further search of depth 2 (this is currently specified):
player1 = "ai+minimax2", player2 = "real", MODEL_NAME_1 = "pretrained", load_from_best_checkpoint_1 = True,
use_window = True, time_delay = 0.3

E.g. 2: playing against the best pretrained AI with deterministic decisions but without further search:
player1 = "ai", player2 = "real", MODEL_NAME_1 = "pretrained", load_from_best_checkpoint_1 = True, 
ai_decision_temperature = 0, use_window = True, time_delay = 0.3

E.g. 3: letting the best pretrained AI play 100 games against minimax with a search depth of 2 (as a fast simulation 
without visualization):
player1 = "ai", player2 = "minimax", MODEL_NAME_1 = "pretrained", load_from_best_checkpoint_1 = True,
ai_decision_temperature = 0, search_depth = 2, use_window = False, n_games = 100
-----------
"""

# choose opponents from "real", "ai", "random", "minimax", "ai+minimax2", '50_minimax_50_random'
player1 = "ai+minimax2"
player2 = 'real'

# if a player is an AI, specify the model name as it was saved e.g. pretrained
# and specify if the best checkpoint should be loaded
MODEL_NAME_1 = "pretrained"
MODEL_NAME_2 = "pretrained"
load_from_best_checkpoint_1 = True
load_from_best_checkpoint_2 = True

# specify ai decision temperature (0 is deterministic, 1 is more random) e.g. 0.1
ai_decision_temperature = 0.1
# if a player is minimax, then specify the search depth
search_depth = 2

# visualization
use_window = True
time_delay = 0.3
n_games = 10


# ------------------------ no need to change anything below this line ------------------------
print("player 1:", player1)
print("player 2:", player2)

nnet1, nnet2 = None, None
if "ai" in player1:
    nnet1 = NeuralNetwork(main_filepath="saves/" + MODEL_NAME_1 + "/gardnerChessAi_", load_from_checkpoint=load_from_best_checkpoint_1)
if "ai" in player2:
    nnet2 = NeuralNetwork(main_filepath="saves/" + MODEL_NAME_2 + "/gardnerChessAi_", load_from_checkpoint=load_from_best_checkpoint_2)

game = Game(player_1_type=player1, player_2_type=player2, window=use_window, time_delay=time_delay,
            ai_decision_temperature=ai_decision_temperature, search_depth=search_depth, nnets=[nnet1, nnet2])
p1_wins, p2_wins, draws, _, average_game_length = game.game_loop(n_games=n_games)

print("player 1 wins:", p1_wins, "\nplayer 2 wins:", p2_wins, "\ndraws:", draws, "\naverage game length:", average_game_length)