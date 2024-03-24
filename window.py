import pygame


class Window:

    def __init__(self, board, scale=0.85):
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_GREY = (130, 130, 130)
        self.COLOR_DARK_GREY = (80, 80, 80)
        self.COLOR_LIGHT_GREEN = (0, 255, 0)
        self.COLOR_BLACK = (0, 0, 0)
        self.board = board
        self.create_window(scale)
        self.load_images()
        self.update_window()

    def create_window(self, scale):
        """
        creates a pygame window
        """
        pygame.init()
        pygame.display.set_caption("Gardner Chess")
        self.board_size = self.board.board_size
        screen_height = pygame.display.Info().current_h
        self.field_pixel_size = int(scale * screen_height / self.board_size)
        self.board_pixel_size = self.field_pixel_size * self.board_size
        self.info_pixel_size = int(self.field_pixel_size / 2)
        self.advantage_bar_pixel_size = int(self.field_pixel_size / 3)
        self.window = pygame.display.set_mode(
            (self.board_pixel_size + self.advantage_bar_pixel_size, self.board_pixel_size + self.info_pixel_size))
        self.font = pygame.font.SysFont('text', int(self.field_pixel_size / 3))
        self.clock = pygame.time.Clock()
        self.FPS = 60

    def load_images(self):
        self.white_square = pygame.transform.scale(pygame.image.load("rsc/white_square.png"),
                                                   (self.field_pixel_size, self.field_pixel_size))
        self.black_square = pygame.transform.scale(pygame.image.load("rsc/black_square.png"),
                                                   (self.field_pixel_size, self.field_pixel_size))
        self.white_pawn = pygame.transform.scale(pygame.image.load("rsc/white_pawn.png"),
                                                 (self.field_pixel_size, self.field_pixel_size))
        self.white_knight = pygame.transform.scale(pygame.image.load("rsc/white_knight.png"),
                                                   (self.field_pixel_size, self.field_pixel_size))
        self.white_bishop = pygame.transform.scale(pygame.image.load("rsc/white_bishop.png"),
                                                   (self.field_pixel_size, self.field_pixel_size))
        self.white_rook = pygame.transform.scale(pygame.image.load("rsc/white_rook.png"),
                                                 (self.field_pixel_size, self.field_pixel_size))
        self.white_queen = pygame.transform.scale(pygame.image.load("rsc/white_queen.png"),
                                                  (self.field_pixel_size, self.field_pixel_size))
        self.white_king = pygame.transform.scale(pygame.image.load("rsc/white_king.png"),
                                                 (self.field_pixel_size, self.field_pixel_size))
        self.black_pawn = pygame.transform.scale(pygame.image.load("rsc/black_pawn.png"),
                                                 (self.field_pixel_size, self.field_pixel_size))
        self.black_knight = pygame.transform.scale(pygame.image.load("rsc/black_knight.png"),
                                                   (self.field_pixel_size, self.field_pixel_size))
        self.black_bishop = pygame.transform.scale(pygame.image.load("rsc/black_bishop.png"),
                                                   (self.field_pixel_size, self.field_pixel_size))
        self.black_rook = pygame.transform.scale(pygame.image.load("rsc/black_rook.png"),
                                                 (self.field_pixel_size, self.field_pixel_size))
        self.black_queen = pygame.transform.scale(pygame.image.load("rsc/black_queen.png"),
                                                  (self.field_pixel_size, self.field_pixel_size))
        self.black_king = pygame.transform.scale(pygame.image.load("rsc/black_king.png"),
                                                 (self.field_pixel_size, self.field_pixel_size))
        self.purple_stars = pygame.transform.scale(pygame.image.load("rsc/purple_stars.gif"),
                                                   (self.field_pixel_size, self.field_pixel_size))

    def update_window(self, current_player=0, winner=0, glitter=(-1, -1), advantage=0):
        """
        updates window
        winner: 0 if game not done, 1 if white won, 2 if black won, 3 if drawn, 4 if unspecified (for spectate mode only)
        glitter: (j-coordinate, i-coordinate) of field for a glitter effect
        advantage: float value between -1 and 1 that shows the advantage: 1: white wins 100%, -1:black wins 100%, 0: even position
        """

        # fill window with light blue background color
        self.window.fill((202, 228, 241))
        """
        draw board
        """
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i + j) % 2 != 0:
                    self.window.blit(self.white_square, (j * self.field_pixel_size, i * self.field_pixel_size))
                else:
                    self.window.blit(self.black_square, (j * self.field_pixel_size, i * self.field_pixel_size))
        """
        draw glitter if needed
        """
        if glitter[0] != -1 != glitter[1]:
            self.window.blit(self.purple_stars,
                             (glitter[0] * self.field_pixel_size, glitter[1] * self.field_pixel_size))
        """
        draw pieces
        """
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board.field[i][j] == self.board.WHITE_PAWN:
                    self.window.blit(self.white_pawn, (j * self.field_pixel_size, i * self.field_pixel_size))
                elif self.board.field[i][j] == self.board.WHITE_KNIGHT:
                    self.window.blit(self.white_knight, (j * self.field_pixel_size, i * self.field_pixel_size))
                elif self.board.field[i][j] == self.board.WHITE_BISHOP:
                    self.window.blit(self.white_bishop, (j * self.field_pixel_size, i * self.field_pixel_size))
                elif self.board.field[i][j] == self.board.WHITE_ROOK:
                    self.window.blit(self.white_rook, (j * self.field_pixel_size, i * self.field_pixel_size))
                elif self.board.field[i][j] == self.board.WHITE_QUEEN:
                    self.window.blit(self.white_queen, (j * self.field_pixel_size, i * self.field_pixel_size))
                elif self.board.field[i][j] == self.board.WHITE_KING:
                    self.window.blit(self.white_king, (j * self.field_pixel_size, i * self.field_pixel_size))
                elif self.board.field[i][j] == self.board.BLACK_PAWN:
                    self.window.blit(self.black_pawn, (j * self.field_pixel_size, i * self.field_pixel_size))
                elif self.board.field[i][j] == self.board.BLACK_KNIGHT:
                    self.window.blit(self.black_knight, (j * self.field_pixel_size, i * self.field_pixel_size))
                elif self.board.field[i][j] == self.board.BLACK_BISHOP:
                    self.window.blit(self.black_bishop, (j * self.field_pixel_size, i * self.field_pixel_size))
                elif self.board.field[i][j] == self.board.BLACK_ROOK:
                    self.window.blit(self.black_rook, (j * self.field_pixel_size, i * self.field_pixel_size))
                elif self.board.field[i][j] == self.board.BLACK_QUEEN:
                    self.window.blit(self.black_queen, (j * self.field_pixel_size, i * self.field_pixel_size))
                elif self.board.field[i][j] == self.board.BLACK_KING:
                    self.window.blit(self.black_king, (j * self.field_pixel_size, i * self.field_pixel_size))
        """
        draw advantage bar
        """
        if winner == 1:
            advantage = 1
        elif winner == 2:
            advantage = -1
        elif winner == 3 or winner == 4:
            advantage = 0
        if advantage < -1:
            advantage = -1
        elif advantage > 1:
            advantage = 1
        advantage = (advantage + 1) / 2  # scale the advantage to the range [0, 1]
        self.window.fill(self.COLOR_BLACK, (self.board_pixel_size, 0, self.advantage_bar_pixel_size,
                                            self.board_pixel_size * (1 - advantage)))
        self.window.fill(self.COLOR_WHITE, (self.board_pixel_size, self.board_pixel_size * (1 - advantage),
                                            self.advantage_bar_pixel_size, self.board_pixel_size * advantage))
        """
        write information under the board
        """
        if winner == 0:
            # show who has the next turn
            font1 = self.font.render('next turn:', True, (130, 130, 130))
            font2 = self.font.render('White' if current_player == 1 else 'Black', True,
                                     (255, 255, 255) if current_player == 1 else (0, 0, 0))
            width_next_turn = int(font1.get_width() * 1.1)
            width_player = font2.get_width()
            width_both = width_next_turn + width_player
            x_start = int(self.board_pixel_size / 2 - width_both / 2)
            self.window.blit(font1, dest=(x_start, int(self.board_pixel_size + 0.3 * self.info_pixel_size)))
            self.window.blit(font2,
                             dest=(x_start + width_next_turn, int(self.board_pixel_size + 0.3 * self.info_pixel_size)))
        else:
            # provide game result information and new game button
            button_text = "new game"
            if winner == 1:
                winner = "white"
                winner_color = self.COLOR_WHITE
            elif winner == 2:
                winner = "black"
                winner_color = self.COLOR_BLACK
            elif winner == 3:
                winner = "draw"
                winner_color = self.COLOR_GREY
            else:
                winner = "unspecified"
                winner_color = self.COLOR_GREY
                button_text = "next game"
            font1 = self.font.render('winner:' if winner != "draw" else "result:", True, self.COLOR_GREY)
            font2 = self.font.render(winner, True, winner_color)
            font3 = self.font.render(button_text, True, self.COLOR_DARK_GREY, self.COLOR_GREY)
            x, y = pygame.mouse.get_pos()
            width_all = int(font1.get_width() * 1.1 + font2.get_width() * 1.5 + font3.get_width())
            x_1 = int((self.board_pixel_size + self.advantage_bar_pixel_size) / 2 - width_all / 2)
            x_2 = x_1 + int(font1.get_width() * 1.1)
            self.new_game_button_x = x_2 + int(font2.get_width() * 1.5)
            if (self.new_game_button_x <= x <= self.new_game_button_x + font3.get_width() and self.board_pixel_size
                    + 0.3 * self.info_pixel_size <= y <= font3.get_height() + self.board_pixel_size + 0.3 * self.info_pixel_size):
                # change color of button if hovered over
                font3 = self.font.render(button_text, True, self.COLOR_LIGHT_GREEN, self.COLOR_GREY)
            self.window.blit(font1, dest=(x_1, self.board_pixel_size + 0.3 * self.info_pixel_size))
            self.window.blit(font2, dest=(x_2, self.board_pixel_size + 0.3 * self.info_pixel_size))
            self.window.blit(font3, dest=(self.new_game_button_x, self.board_pixel_size + 0.3 * self.info_pixel_size))
        # update display
        self.clock.tick(self.FPS)
        pygame.display.update()

    def wait_for_action(self, legal_moves, current_player, advantage):
        """
        returns the action tuple of the two fields, on which the current player clicked (if they are legal)
        format: (from_j, from_i, to_j, to_i)
        """
        while True:
            # choosing the 'from' coordinates
            for event1 in pygame.event.get():
                if event1.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event1.type == pygame.MOUSEBUTTONUP and event1.button == 1:  # left click
                    from_j, from_i = pygame.mouse.get_pos()
                    from_j = int(from_j / self.field_pixel_size)
                    from_i = int(from_i / self.field_pixel_size)
                    if 0 <= from_i <= 4 and 0 <= from_j <= 4:
                        for move in legal_moves:
                            if from_j == move[0] and from_i == move[1]:  # piece has a legal move
                                self.update_window(current_player=current_player, glitter=(from_j, from_i),
                                                   advantage=advantage)
                                break
                        run = True
                        while run:
                            # choosing the 'to' coordinates
                            for event2 in pygame.event.get():
                                if event2.type == pygame.QUIT:
                                    pygame.quit()
                                    return
                                elif event2.type == pygame.MOUSEBUTTONUP and event2.button == 3:  # right click: restart choosing 'from' coordinates
                                    self.update_window(current_player=current_player, advantage=advantage)
                                    run = False
                                    break
                                elif event2.type == pygame.MOUSEBUTTONUP and event2.button == 1:  # left click
                                    to_j, to_i = pygame.mouse.get_pos()
                                    to_j = int(to_j / self.field_pixel_size)
                                    to_i = int(to_i / self.field_pixel_size)
                                    if 0 <= to_j <= 4 and 0 <= to_i <= 4:
                                        for move in legal_moves:
                                            if from_j == move[0] and from_i == move[1] and to_j == move[2] and to_i == \
                                                    move[3]:  # piece has this legal move
                                                return move
                                        # they can choose another piece
                                        from_j_if_not_empty_field, from_i_if_not_empty_field = to_j, to_i
                                        for move in legal_moves:
                                            if from_j_if_not_empty_field == move[0] and from_i_if_not_empty_field == \
                                                    move[1]:  # piece has a legal move
                                                from_j, from_i = from_j_if_not_empty_field, from_i_if_not_empty_field
                                                self.update_window(current_player=current_player,
                                                                   glitter=(from_j, from_i), advantage=advantage)
                                                break

    def check_if_new_game(self):
        """
        waits for the player to click on 'new game'
        animates the 'new game' button if hovered over
        returns True afterward
        """
        while True:
            for event in pygame.event.get():
                font = self.font.render('new game', True, self.COLOR_LIGHT_GREEN, self.COLOR_GREY)
                x, y = pygame.mouse.get_pos()
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif not (self.new_game_button_x <= x <= self.new_game_button_x + font.get_width() and self.board_pixel_size
                          + 0.3 * self.info_pixel_size <= y <= font.get_height() + self.board_pixel_size + 0.3 * self.info_pixel_size):
                    # not hovered over button
                    font = self.font.render('new game', True, self.COLOR_DARK_GREY, self.COLOR_GREY)
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:  # 1: left click on button
                    return
                self.window.blit(font, dest=(self.new_game_button_x, self.board_pixel_size + 0.3 * self.info_pixel_size))
                self.clock.tick(self.FPS)
                pygame.display.update()

    def spectate(self, game_states, ai_color="white", opponent="minimax"):
        print("spectating ai (" + ai_color + ") against", opponent)
        color_index = 0 if ai_color == "white" else 1
        current_player = color_index + 1
        if opponent == "minimax":
            opponent_index = 0
        elif opponent == "random":
            opponent_index = 1
        elif opponent == "50_minimax_50_random":
            opponent_index = 2
        elif opponent == "90_minimax_10_random":
            opponent_index = 3
        else:
            print("opponent not found")
            return
        if len(game_states[opponent_index]) == 0:
            print("no episodes to spectate!")
            print("(the model has not trained 100 yet)")
            return
        game_state_index = 0
        episode_index = 0
        print("Episode:", 100 * (episode_index + 1))
        self.update_window(current_player=current_player, winner=4)
        font = self.font.render('next game', True, self.COLOR_DARK_GREY, self.COLOR_GREY)
        while True:
            for event in pygame.event.get():
                x, y = pygame.mouse.get_pos()
                self.board.field = game_states[opponent_index][episode_index][color_index][game_state_index]
                if event.type == pygame.QUIT:
                    pygame.quit()
                elif (self.new_game_button_x <= x <= self.new_game_button_x + font.get_width() and self.board_pixel_size
                      + 0.3 * self.info_pixel_size <= y <= font.get_height() + self.board_pixel_size + 0.3 * self.info_pixel_size):
                    if event.type == pygame.MOUSEBUTTONUP and event.button == 1:  # 1: left click
                        if episode_index == len(game_states[opponent_index]) - 1:
                            print("this was the last episode that can be spectated!")
                        else:
                            game_state_index = 0
                            episode_index += 1
                            if episode_index >= len(game_states[opponent_index]):
                                episode_index = len(game_states[opponent_index]) - 1
                            print("Episode:", 100 * (episode_index + 1))
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    if game_state_index > 0:
                        game_state_index -= 1
                elif keys[pygame.K_RIGHT]:
                    if game_state_index < len(game_states[opponent_index][episode_index][color_index]) - 1:
                        game_state_index += 1
                self.update_window(current_player=current_player, winner=4)
