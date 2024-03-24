import copy
import random
import time
import numpy as np
import pygame

from board import Board


class Game:

    def __init__(self, player_1_type='real', player_2_type='real', window=False, search_depth=4,
                 nnets=(None, None), time_delay=0, ai_decision_temperature=0.05, window_scale=0.84):
        self.board = Board()
        self.use_window = window
        if player_1_type == "real" or player_2_type == "real":
            self.print_rules()
            self.use_window = True
        self.search_depth = search_depth
        self.nnets = nnets
        self.ai_decision_temperature = ai_decision_temperature
        self.time_delay = time_delay
        self.player_1_type = player_1_type
        self.player_2_type = player_2_type
        self.white_player_type = self.player_1_type
        self.black_player_type = self.player_2_type
        self.WHITE_PLAYER = 1
        self.BLACK_PLAYER = 2
        self.player_1_color = self.WHITE_PLAYER
        self.player_2_color = self.BLACK_PLAYER
        self.current_player = self.WHITE_PLAYER
        self.current_player_type = self.white_player_type
        # game ends in a draw if both players don't capture pieces for 10 moves
        self.moves_without_captured_piece_and_without_pawn_moves = 0
        if self.use_window:
            from window import Window
            self.window = Window(self.board, scale=window_scale)
            self.window.update_window(current_player=self.current_player)
        else:
            self.time_delay = 0

    def game_loop(self, n_games=1):
        p1_wins = 0
        p2_wins = 0
        draws = 0
        board_states_of_a_game = [[copy.copy(self.board.field)], [copy.copy(self.board.field)]]
        n_moves = 0
        n_games_to_play = n_games
        while n_games_to_play != 0:
            winner = 0
            while winner == 0:
                move = self.get_move()
                winner = self.do_move(move)
                n_moves += 1
                if n_games_to_play == 1 or n_games_to_play == 2:
                    if self.player_1_color == self.WHITE_PLAYER:
                        board_states_of_a_game[0].append(copy.copy(self.board.field))
                    else:
                        board_states_of_a_game[1].append(copy.copy(self.board.field))
            if self.use_window and self.player_1_type != "ai_evaluation":
                self.window.check_if_new_game()
            if winner == self.player_1_color:
                p1_wins += 1
            elif winner == self.player_2_color:
                p2_wins += 1
            elif winner == 3:
                draws += 1
            n_games_to_play -= 1
            self.start_new_game()
        average_game_length = round(n_moves / n_games, 1)
        return p1_wins, p2_wins, draws, board_states_of_a_game, average_game_length

    def print_rules(self):
        print("----------------------------------")
        print("---------------Rules---------------")
        print("----------------------------------")
        print("same rules as in normal chess except:")
        print("5*5 board")
        print("you are not forced to save yourself from check ...")
        print("... but it is wise --> otherwise, your king can be captured")
        print("no double pawn move")
        print("no en passant")
        print("no castling")
        print("pawns can only promote to a queen")
        print("----------------------------------")

    def print_game_result_info_in_console(self, winner):
        print("----------------------------------")
        if winner == self.WHITE_PLAYER:
            print("Weiß hat gewonnen!")
        elif winner == self.BLACK_PLAYER:
            print("Schwarz hat gewonnen!")
        elif winner == 3:
            print("Unentschieden, da 20 Züge lang keine Einheit geschlagen wurde!")
        print("----------------------------------")

    def start_new_game(self):
        # reset countable variables
        self.current_player = self.WHITE_PLAYER
        # switch who starts
        old_white_player_type = self.white_player_type
        old_black_player_type = self.black_player_type
        self.white_player_type = old_black_player_type
        self.black_player_type = old_white_player_type
        self.player_1_color = 3 - self.player_1_color
        self.player_2_color = 3 - self.player_2_color

        self.current_player_type = self.white_player_type
        self.moves_without_captured_piece_and_without_pawn_moves = 0
        # reset board
        self.board.reset_board()
        if self.use_window:
            self.window.update_window(current_player=self.current_player)

    def get_piece_positions_of_player(self, field=None, player=None):
        """
        returns the positions of all the pieces of the player
        uses self.current_player when no player is provided and self.board.field array when no field array is provided
        """
        if field is None:
            field = self.board.field
        if player == None:
            player = self.current_player
        piece_positions = []
        for i in range(self.board.board_size):
            for j in range(self.board.board_size):
                piece = field[i][j]
                if (player == self.WHITE_PLAYER and piece in self.board.WHITE_PIECES) or (player == self.BLACK_PLAYER and piece in self.board.BLACK_PIECES):
                    piece_positions.append([j, i])
        return piece_positions

    def get_all_legal_moves(self, current_player_piece_positions, field=None):
        """
        refers to self.board.field array when no other field array is provided
        returns a list of all legal moves in the format [(from_j, from_i, to_j, to_i), (...), ...]
        """

        if len(current_player_piece_positions) == 0:
            print("Verloren, keine Einheiten mehr")
        else:
            legal_moves = []
            WHITE_PIECES = self.board.WHITE_PIECES
            BLACK_PIECES = self.board.BLACK_PIECES
            EMPTY_FIELD = self.board.EMPTY_FIELD

            if field is None:
                field = self.board.field
            for piece in range(len(current_player_piece_positions)):
                from_j, from_i = current_player_piece_positions[piece][0], current_player_piece_positions[piece][1]
                piece_type = field[from_i][from_j]

                if piece_type == self.board.WHITE_PAWN:
                    if from_i - 1 >= 0:
                        if from_j - 1 >= 0:
                            if field[from_i - 1][from_j - 1] in BLACK_PIECES:
                                legal_moves.append((from_j, from_i, from_j - 1, from_i - 1))
                        if from_j + 1 <= 4:
                            if field[from_i - 1][from_j + 1] in BLACK_PIECES:
                                legal_moves.append((from_j, from_i, from_j + 1, from_i - 1))
                        if field[from_i - 1][from_j] == EMPTY_FIELD:
                            legal_moves.append((from_j, from_i, from_j, from_i - 1))

                elif piece_type == self.board.BLACK_PAWN:
                    if from_i + 1 <= 4:
                        if from_j - 1 >= 0:
                            if field[from_i + 1][from_j - 1] in WHITE_PIECES:
                                legal_moves.append((from_j, from_i, from_j - 1, from_i + 1))
                        if from_j + 1 <= 4:
                            if field[from_i + 1][from_j + 1] in WHITE_PIECES:
                                legal_moves.append((from_j, from_i, from_j + 1, from_i + 1))
                        if field[from_i + 1][from_j] == EMPTY_FIELD:
                            legal_moves.append((from_j, from_i, from_j, from_i + 1))

                elif piece_type == self.board.WHITE_KNIGHT or piece_type == self.board.BLACK_KNIGHT:
                    for k in ((-1, -2), (-1, 2), (1, -2), (1, 2), (-2, -1), (-2, 1), (2, -1), (2, 1)):
                        if 0 <= from_j + k[0] <= 4 and 0 <= from_i + k[1] <= 4:  # if on board
                            if piece_type == self.board.WHITE_KNIGHT and field[from_i + k[1]][from_j + k[0]] not in WHITE_PIECES:
                                legal_moves.append((from_j, from_i, from_j + k[0], from_i + k[1]))
                            elif piece_type == self.board.BLACK_KNIGHT and field[from_i + k[1]][from_j + k[0]] not in BLACK_PIECES:
                                legal_moves.append((from_j, from_i, from_j + k[0], from_i + k[1]))

                elif (piece_type == self.board.WHITE_BISHOP or piece_type == self.board.BLACK_BISHOP or piece_type == self.board.WHITE_QUEEN
                      or piece_type == self.board.BLACK_QUEEN or piece_type == self.board.WHITE_ROOK or piece_type == self.board.BLACK_ROOK):
                    if (piece_type == self.board.WHITE_BISHOP or piece_type == self.board.BLACK_BISHOP or
                            piece_type == self.board.WHITE_QUEEN or piece_type == self.board.BLACK_QUEEN):
                        for direktion in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
                            for distance in range(1, 5):
                                if 0 <= from_j + direktion[0] * distance <= 4 and 0 <= from_i + direktion[1] * distance <= 4:  # if on board
                                    if (piece_type in WHITE_PIECES and field[from_i + direktion[1] * distance][from_j + direktion[0] * distance]
                                            not in WHITE_PIECES):  # if own white pieces aren't in the way
                                        legal_moves.append((from_j, from_i, from_j + direktion[0] * distance, from_i + direktion[1] * distance))
                                        if field[from_i + direktion[1] * distance][from_j + direktion[0] * distance] != EMPTY_FIELD:  # if opponent piece is captured
                                            break
                                    elif (piece_type in BLACK_PIECES and field[from_i + direktion[1] * distance][from_j + direktion[0] * distance]
                                          not in BLACK_PIECES):  # if own black pieces aren't in the way
                                        legal_moves.append((from_j, from_i, from_j + direktion[0] * distance, from_i + direktion[1] * distance))
                                        if field[from_i + direktion[1] * distance][
                                            from_j + direktion[0] * distance] != EMPTY_FIELD:  # if opponent piece is captured
                                            break
                                    else:
                                        break
                                else:
                                    break
                    if (piece_type == self.board.WHITE_ROOK or piece_type == self.board.BLACK_ROOK or
                            piece_type == self.board.WHITE_QUEEN or piece_type == self.board.BLACK_QUEEN):
                        for direktion in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                            for distance in range(1, 5):
                                if 0 <= from_j + direktion[0] * distance <= 4 and 0 <= from_i + direktion[1] * distance <= 4:  # if on board
                                    if (piece_type in WHITE_PIECES and field[from_i + direktion[1] * distance][from_j + direktion[0] * distance]
                                            not in WHITE_PIECES):  # if own white pieces aren't in the way
                                        legal_moves.append((from_j, from_i, from_j + direktion[0] * distance, from_i + direktion[1] * distance))
                                        if field[from_i + direktion[1] * distance][from_j + direktion[0] * distance] != EMPTY_FIELD:  # if opponent piece is captured
                                            break
                                    elif (piece_type in BLACK_PIECES and field[from_i + direktion[1] * distance][from_j + direktion[0] * distance]
                                          not in BLACK_PIECES):  # if own black pieces aren't in the way
                                        legal_moves.append((from_j, from_i, from_j + direktion[0] * distance, from_i + direktion[1] * distance))
                                        if field[from_i + direktion[1] * distance][from_j + direktion[0] * distance] != EMPTY_FIELD:  # if opponent piece is captured
                                            break
                                    else:
                                        break
                                else:
                                    break
                elif piece_type == self.board.WHITE_KING or piece_type == self.board.BLACK_KING:
                    for step in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                        if 0 <= from_j + step[0] <= 4 and 0 <= from_i + step[1] <= 4:  # if on board
                            if piece_type == self.board.WHITE_KING and field[from_i + step[1]][from_j + step[0]] not in WHITE_PIECES:  # if own white pieces aren't in the way
                                legal_moves.append((from_j, from_i, from_j + step[0], from_i + step[1]))
                            elif piece_type == self.board.BLACK_KING and field[from_i + step[1]][
                                from_j + step[0]] not in BLACK_PIECES:  # if own white pieces aren't in the way
                                legal_moves.append((from_j, from_i, from_j + step[0], from_i + step[1]))
            return legal_moves

    def get_move(self):
        """
        returns the move decision of the current player
        """
        current_player_piece_positions = self.get_piece_positions_of_player()
        legal_moves = self.get_all_legal_moves(current_player_piece_positions)
        if self.current_player_type == "real":
            move = self.window.wait_for_action(legal_moves, self.current_player, self.get_white_advantage(2))
        elif self.current_player_type == "minimax":
            value, move = self.alpha_beta_pruning(self.board.field, self.search_depth, self.search_depth)
        elif self.current_player_type == "minimax+search":
            value, move = self.alpha_beta_pruning(self.board.field, self.search_depth, self.search_depth,
                                                  search_best_nodes_only=True, n_best_nodes='auto', add_noise=True,
                                                  noise_factor=0.05)
        elif self.current_player_type == "50_minimax_50_random":
            if random.random() < 0.5:
                move = self.get_random_move()
            else:
                value, move = self.alpha_beta_pruning(self.board.field, self.search_depth, self.search_depth)
        elif self.current_player_type == "90_minimax_10_random":
            if random.random() < 0.1:
                move = self.get_random_move()
            else:
                value, move = self.alpha_beta_pruning(self.board.field, self.search_depth, self.search_depth)
        elif self.current_player_type == "ai" or self.current_player_type == "ai_evaluating":
            move = self.get_ai_move(legal_moves, temperature=self.ai_decision_temperature)
        elif self.current_player_type == "ai+minimax2":
            value, move = self.alpha_beta_pruning(self.board.field, 2, 2,
                                                  evaluation_mode="ai", add_noise=False, noise_factor=0.2)
        elif self.current_player_type == "ai+search":
            value, move = self.alpha_beta_pruning(self.board.field, self.search_depth, self.search_depth,
                                                  evaluation_mode="ai", search_best_nodes_only=True,
                                                  n_best_nodes='auto', add_noise=False, noise_factor=0.05)
        else:  # random opponent
            move = self.get_random_move()
        return move

    def get_random_move(self):
        """
        returns a random move for the current player
        """
        moves = self.get_all_legal_moves(self.get_piece_positions_of_player())
        return moves[random.randint(0, len(moves) - 1)]

    def get_ai_move(self, legal_moves, temperature):
        """
        returns a move that the AI would pick
        temperature:
        small --> confident and deterministic decisions (plays with "best" strategy)
        high --> unconfident and more random decisions
        """
        current_player_index = self.current_player - 1 if self.player_1_color == self.WHITE_PLAYER else (3 - self.current_player) - 1
        reachable_boards = self.get_all_reachable_boards(legal_moves, reverse=True if self.current_player == 2 else False)
        nnet = self.nnets[current_player_index]
        nnet_input_of_new_possible_boards = nnet.get_nnet_input(reachable_boards)
        q_values_of_new_possible_boards = nnet.get_q_values(nnet_input_of_new_possible_boards)
        if temperature != 0:
            nnet_input_of_current_board = nnet.get_nnet_input(np.array([self.board.field]))
            move, _ = nnet.get_probabilistic_move(legal_moves, q_values_of_new_possible_boards, nnet_input_of_current_board, temperature=temperature)
            return move
        else:  # deterministic move if temperature == 0
            return legal_moves[np.argmax(q_values_of_new_possible_boards)]

    def do_move(self, move):
        """
        changes the board according to the move and prepares next move
        """
        from_j, from_i, to_j, to_i = move[0], move[1], move[2], move[3]
        piece_type = self.board.field[from_i][from_j]

        self.moves_without_captured_piece_and_without_pawn_moves += 1
        if piece_type == self.board.WHITE_PAWN or piece_type == self.board.BLACK_PAWN:
            self.moves_without_captured_piece_and_without_pawn_moves = 0
            # promote to queen if pawn reaches the other side
            if to_i == 0:
                piece_type = self.board.WHITE_QUEEN
            elif to_i == 4:
                piece_type = self.board.BLACK_QUEEN
        elif self.board.field[to_i][to_j] != self.board.EMPTY_FIELD:
            self.moves_without_captured_piece_and_without_pawn_moves = 0

        self.board.field[from_i][from_j] = 0
        self.board.field[to_i][to_j] = piece_type

        self.current_player = 3 - self.current_player  # switch player turns
        winner = self.check_if_game_is_over()
        if self.use_window:
            if self.current_player_type != "real":
                pygame.event.get()
                if self.time_delay != 0:
                    time.sleep(self.time_delay)
            self.window.update_window(current_player=self.current_player, winner=winner, advantage=self.get_white_advantage(2))
        self.current_player_type = self.white_player_type if self.current_player == self.WHITE_PLAYER else self.black_player_type
        return winner

    def check_if_game_is_over(self, field=None):
        """
        uses self.board.field array when no field array is provided
        returns:
        0 if game isn't over
        1 if self.WHITE_PLAYER won
        2 if self.BLACK_PLAYER won
        3 if draw
        """
        if field is None:
            field = self.board.field
        if self.board.BLACK_KING not in field:
            return 1
        elif self.board.WHITE_KING not in field:
            return 2
        elif self.moves_without_captured_piece_and_without_pawn_moves >= 20:
            return 3
        else:
            return 0

    def get_all_reachable_boards(self, moves, reverse=False, field=None):
        """
        returns all possible boards which can be reached with one move in a numpy array
        if reverse = True, moves + board including pieces colors will be reversed
        if field = None, self.board.field will be used
        """
        if field is None:
            field = self.board.field
        new_boards = []
        for move in moves:
            if reverse:
                new_board = self.board.get_reversed_board_view()  # flip board and pieces if reversed
            else:
                new_board = copy.deepcopy(field)
            if reverse:
                from_j, from_i, to_j, to_i = move[0], 4 - move[1], move[2], 4 - move[3]  # flip moves if reversed
            else:
                from_j, from_i, to_j, to_i = move[0], move[1], move[2], move[3]
            piece_type = field[from_i][from_j]
            if piece_type == self.board.WHITE_PAWN and to_i == 0:
                piece_type = self.board.WHITE_QUEEN
            elif piece_type == self.board.BLACK_PAWN and to_i == 4:
                piece_type = self.board.BLACK_QUEEN
            new_board[from_i][from_j] = 0
            new_board[to_i][to_j] = piece_type
            new_boards.append(new_board)
        return np.array(new_boards)

    def alpha_beta_pruning(self, node, depth, starting_depth, a=-1000, b=1000, maximizing_player=True, evaluation_mode="material",
                           search_best_nodes_only=False, n_best_nodes='auto', add_noise=False, noise_factor=0.05):
        """
        alpha beta pruning achieves same results as minimax, but with timewise optimization
        search_best_nodes_only with n_best_nodes and add_noise with noise_factor modify the vanilla alpha beta pruning,
        when search_best_nodes_only=False and add_noise=False (or nothing specified), it is the vanilla alpha beta pruning
        """
        current_search_player = self.current_player if maximizing_player else 3 - self.current_player
        # check if terminal node
        winner = self.check_if_game_is_over(node)
        if winner != 0:
            terminal_node = True
        else:
            terminal_node = False
        # return value of board if terminal node
        if depth == 0 or terminal_node:
            if evaluation_mode == "material":
                white_material_value, black_material_value = self.board.get_material_value(node)
                if self.current_player == self.WHITE_PLAYER:
                    worth_of_game_state = white_material_value - black_material_value
                else:
                    worth_of_game_state = black_material_value - white_material_value
                return worth_of_game_state + random.random()
            elif evaluation_mode == "ai":
                current_player_index = self.current_player - 1 if self.player_1_color == self.WHITE_PLAYER else (3 - self.current_player) - 1
                nnet = self.nnets[current_player_index]
                # white and black are exchanged because it needs to be viewed by the search_player from before
                if current_search_player == self.WHITE_PLAYER:
                    worth_of_game_state = nnet.get_q_values(nnet.get_nnet_input(np.array([self.board.get_reversed_board_view(field=node)])))[0]
                else:
                    worth_of_game_state = nnet.get_q_values(nnet.get_nnet_input(np.array([node])))[0]
                if add_noise:
                    worth_of_game_state += noise_factor * worth_of_game_state * random.random()
                return worth_of_game_state if not maximizing_player else worth_of_game_state * -1
        # get new possible boards
        moves = self.get_all_legal_moves(self.get_piece_positions_of_player(field=node, player=current_search_player), node)
        child_nodes = self.get_all_reachable_boards(moves, field=node)
        if search_best_nodes_only:
            # get child value
            child_priorities = []
            if evaluation_mode == "material":
                for i in range(len(child_nodes)):
                    white_material_value, black_material_value = self.board.get_material_value(child_nodes[i])
                    if current_search_player == self.WHITE_PLAYER:
                        child_priority = white_material_value - black_material_value
                    else:
                        child_priority = black_material_value - white_material_value
                    child_priorities.append(child_priority)
            elif evaluation_mode == "ai":
                current_player_index = self.current_player - 1 if self.player_1_color == self.WHITE_PLAYER else (3 - self.current_player) - 1
                nnet = self.nnets[current_player_index]
                if current_search_player == self.BLACK_PLAYER:
                    reversed_board_views_of_child_nodes = []
                    for child in child_nodes:
                        reversed_board_views_of_child_nodes.append(self.board.get_reversed_board_view(field=child))
                    reversed_board_views_of_child_nodes = np.array(reversed_board_views_of_child_nodes)
                    child_priorities = nnet.get_q_values(nnet.get_nnet_input(reversed_board_views_of_child_nodes))
                else:
                    child_priorities = nnet.get_q_values(nnet.get_nnet_input(child_nodes))
            if add_noise:
                for i in range(len(child_priorities)):
                    child_priorities[i] += noise_factor * random.random() * child_priorities[i]
            best_child_nodes = []
            best_moves = []
            if n_best_nodes == 'auto':
                n_best_nodes = int(len(moves) / 2 + 1)
                if n_best_nodes > 8:
                    n_best_nodes = 8
            for j in range(n_best_nodes):
                if evaluation_mode == "material":
                    max_value_index = child_priorities.index(max(child_priorities))
                elif evaluation_mode == "ai":
                    max_value_index = np.argmax(child_priorities)
                best_child_nodes.append(child_nodes[max_value_index])
                best_moves.append(moves[max_value_index])
                child_priorities[max_value_index] = -9999  # needs to be smaller than maximum negative value
            child_nodes = best_child_nodes
            moves = best_moves
        if depth == starting_depth:
            best_move = None
        index = -1
        if maximizing_player:
            value = -300  # smaller than reachable
            best_value = value
            for child in child_nodes:
                index += 1
                value = max(value,
                            self.alpha_beta_pruning(child, depth - 1, starting_depth, a, b, False,
                                                    evaluation_mode, search_best_nodes_only, n_best_nodes))
                if value > b:
                    break  # β cutoff
                a = max(a, value)
                if value > best_value:
                    best_move = moves[index]
                    best_value = value
            if depth == starting_depth:
                return value, best_move
            else:
                return value
        else:
            value = 300  # bigger than reachable
            best_value = value
            for child in child_nodes:
                index += 1
                if child is not None:
                    value = min(value, self.alpha_beta_pruning(child, depth - 1, starting_depth, a, b, True,
                                                               evaluation_mode, search_best_nodes_only, n_best_nodes))
                    if value < a:
                        break  # α cutoff
                    b = min(b, value)
                    if value > best_value:
                        best_move = moves[index]
                        best_value = value
            if depth == starting_depth:
                return value, best_move
            else:
                return value

    def get_white_advantage(self, depth):
        """
        returns the white advantage over black
        advantage: float value between -1 and 1 that shows the advantage: 1: white wins 100%, -1:black wins 100%, 0: even position
        """
        if self.white_player_type == self.black_player_type == "ai_training" or self.white_player_type == "ai_evaluating":
            return 0  # don't calculate it when AI is training or evaluating
        winner = self.check_if_game_is_over()
        if winner == 1:
            advantage = 1
        elif winner == 2:
            advantage = -1
        elif winner == 3:
            advantage = 0
        else:
            value, advantage_move = self.alpha_beta_pruning(self.board.field, depth, depth)
            advantage_board = self.get_all_reachable_boards([advantage_move], field=copy.copy(self.board.field))
            white_material_value, black_material_value = self.board.get_material_value(advantage_board[0])
            if white_material_value < 0:
                advantage = -1
            elif black_material_value < 0:
                advantage = 1
            else:
                advantage = 2 * white_material_value / (white_material_value + black_material_value + 0.000001) - 1
        return advantage
