from gym import spaces
import numpy as np

NUM_ACTIONS = (25) + (25 * 4) + (25 * 4 * 25) # 2625 possible actions
BOARD_DIM = 5
NUM_PIECES_PER_PLAYER = 12
MAX_GAME_LENGTH = 100

class Choko_Env:
    def __init__(self):
        self.obs_space = spaces.Box(low = 0, high = 2, shape = (BOARD_DIM, BOARD_DIM), dtype = np.int8)
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.reset()
    
    def reset(self):
        self.board = np.zeros((BOARD_DIM, BOARD_DIM), dtype = np.int8)
        self.player = 1
        self.drop_initiative = 1
        self.pieces_captured = {1: 0, 2: 0} # {1: # of player 1's pieces captured, 2: # of player 2's pieces captured}
        # pieces_left = {1: # of player 1's pieces left to place, 2: # of player 2's pieces left to place}
        self.pieces_left = {1: NUM_PIECES_PER_PLAYER, 2: NUM_PIECES_PER_PLAYER}
        self.freeze_turns = False
        self.num_moves = 0
        return self.fetch_obs_action_mask()
    
    def step(self, action):
        '''
        Given an action, performs the action on the board and returns a tuple of
        (state, reward, done, info). Note that done is a string that can be "won", "draw", or "ongoing".
        '''
        assert self.action_space.contains(action), f"Invalid action {action}"

        # First 25 moves are for placing a piece
        info_map = self.to_info(action)
        move_type = info_map["move_type"]
        if move_type == "place":
            if self.drop_initiative == 0:
                # this is the first placement after a moving phase
                self.drop_initiative = self.player

            if self.pieces_left[self.player] == 0:
                raise ValueError("Invalid move: No pieces left to place")
            row, col = info_map["row"], info_map["col"]
            if self.board[row, col] == 0:
                self.board[row, col] = self.player
                self.pieces_left[self.player] -= 1
            else:
                raise ValueError("Invalid move: Cell already occupied")
        
        # next 48 moves are for moving a piece
        elif move_type == "move":
            
            if self.drop_initiative != self.player and self.drop_initiative != 0:
                # we are trying to move a piece but the drop initiative is set
                # to the other player
                raise ValueError("Invalid move: Not your drop initiative")
            
            self.drop_initiative = 0
            action -= 25
            index, direction = divmod(action, 4)
            row, col = divmod(index, 5)

            row = info_map["row"]
            col = info_map["col"]
            direction = info_map["direction"]
            
            next_row, next_col, _, _ = self.check_valid_move(row, col, direction, move_type)
            self.board[next_row, next_col] = self.player
            self.board[row, col] = 0
        
        else:
            if self.drop_initiative != self.player and self.drop_initiative != 0:
                # we are trying to move a piece but the drop initiative is set
                # to the other player
                raise ValueError("Invalid move: Not your drop initiative")
            
            self.drop_initiative = 0
            capture_row, capture_col = info_map["capture_row"], info_map["capture_col"]
            og_row, og_col, direction = info_map["row"], info_map["col"], info_map["direction"]

            next_row, next_col, between_row, between_col = self.check_valid_move(og_row, og_col, direction, move_type)

            # perform the jump
            self.board[next_row, next_col] = self.player
            self.board[og_row, og_col] = 0

            # first capture:
            self.board[between_row, between_col] = 0
            self.pieces_captured[3 - self.player] += 1

            # extra capture
            if self.board[capture_row, capture_col] == self.player:
                raise ValueError("Invalid move: Cannot capture your own piece")
            
            if self.board[capture_row, capture_col] != 0:
                # we are capturing an opponent's piece
                self.board[capture_row, capture_col] = 0
                self.pieces_captured[3 - self.player] += 1
        
        if not self.freeze_turns:
            self.player = 3 - self.player
        
        self.num_moves += 1
        
        state: tuple[np.ndarray, np.ndarray] = self.fetch_obs_action_mask()
        _, mask = state
        game_condition: str = self.evaluate_termination(mask)
        
        if game_condition == "draw" or game_condition == "won":
            done = True
        else:
            done = False
        
        if game_condition != "ongoing":
            if game_condition == "draw":
                reward = 0
            else:
                reward = 1
        else:
            reward = 0
        done: str = self.evaluate_termination(mask)
        reward: int = 1 if done else 0 # reward is 1 if the player playing wins, 0 otherwise
        info = {}

        return state, reward, done, info
    
    def evaluate_termination(self, mask) -> str:
        '''
        Evaluates whether the game is won, drawn, or ongoing.
        Returns "won" if the player who is currently playing has won,
        "draw" if the game is a draw, and "ongoing" if the game is still ongoing.
        '''
        if self.num_moves >= MAX_GAME_LENGTH:
            return "draw"
        
        if mask.sum() == 0:
            return "draw"

        if self.pieces_captured[3 - self.player] == NUM_PIECES_PER_PLAYER:
            # I want to evaluate if the player who is currently playing
            # has captured all the opponent's pieces
            return "won"
        
        return "ongoing"
    
    def fetch_obs_action_mask(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Begin with all actions impossible. For each piece on the board,
        find out if the piece is my piece, and then use helper function
        to find all the ways I can move this piece.

        For captures, it will be helpful to have an array of all
        opponents pieces. 
        '''
        obs = self.board.flatten()
        # get a list of the indices of all the opponent's pieces on the board
        opp_caps = self.get_opponent_pieces()

        mask = np.zeros(NUM_ACTIONS, dtype = np.int8)
        for row in range(BOARD_DIM):
            for col in range(BOARD_DIM):
                self.get_valid_moves(row, col, opp_caps, mask)
        return (obs, mask)
                
    
    def get_valid_moves(self, row, col, opp_caps, mask) -> None:
        '''
        given a row and column, validates the moves in the passed in mask
        # TODO, test this a bit

        returns nothing but modifies the mask in place. 
        '''
        # first placement
        if self.board[row, col] == 0: # needs to be an empty space
            if self.pieces_left[self.player] > 0:
                action = self.to_action(row=row, col=col, move_type="place")
                mask[action] = 1
        
        # normal move
        if ((self.drop_initiative == self.player or self.drop_initiative == 0) 
            and self.board[row, col] == self.player):
            for direction in range(4):
                if self.is_move_valid(row, col, direction, move_type="move"):
                    action = self.to_action(row=row, col=col, move_type="move", direction=direction)
                    mask[action] = 1
                if self.is_move_valid(row, col, direction, move_type="jump"):
                    # we can perform a jump here too. Still need each additional
                    # extra capture piece
                    action = self.to_action(
                        row = row,
                        col = col,
                        move_type = "jump",
                        direction = direction,
                        capture_row = 0,
                        capture_col = 0
                    )
                    _, _, between_row, between_col = self.check_valid_move(row, col, direction, move_type="jump")
                    for capture in opp_caps:
                        if capture != between_row * BOARD_DIM + between_col:
                            # checking that we the extra capture is not the same as the
                            # piece we are jumping over
                            mask[action + capture] = 1

        return mask

    def get_opponent_pieces(self):
        '''
        returns a list of the indices of the opponent's pices on the board.
        
        Note that the indices are in the format of row * BOARD_DIM + col.
        ''' 
        indices = []
        for row in range(BOARD_DIM):
            for col in range(BOARD_DIM):
                if self.board[row, col] == 3 - self.player:
                    indices.append(row * BOARD_DIM + col)
        
        # shitty debugging to make sure the other code works, TODO switch later to the vectorized code.
        opp = 3 - self.player
        flat = self.board.ravel()
        first_list = np.flatnonzero(flat == opp).tolist()
        assert first_list == indices
        return indices
    
    def is_move_valid(self, row, col, direction, move_type, player=None):
        '''
        returns True is the move is valid, False otherwise.
        
        Same thing as check_valid_move, but does not return the next row and col
        information, just a boolean.
        '''
        player = self.player if player is None else player

        if self.board[row, col] != player:
            # Cannot move a piece that is not yours
            return False
        next_row = row
        next_col = col
        between_row = row
        between_col = col
        # 0, 1, 2, 3 = up, right, down, left
        if move_type == "jump":
            if direction == 0:
                next_row = row - 2
                between_row = row - 1
            elif direction == 2:
                next_row = row + 2
                between_row = row + 1
            elif direction == 1:
                next_col = col + 2
                between_col = col + 1
            elif direction == 3:
                next_col = col - 2
                between_col = col - 1
        else:
            # Normal move
            if direction == 0:
                next_row = row - 1
            elif direction == 2:
                next_row = row + 1
            elif direction == 1:
                next_col = col + 1
            elif direction == 3:
                next_col = col - 1
        
        if next_row < 0 or next_row >= BOARD_DIM or next_col < 0 or next_col >= BOARD_DIM:
            # out of bounds
            return False
        
        if self.board[next_row, next_col] != 0:
            # cell we are moving to is already occupied
            return False

        if move_type == "jump":
            if self.board[between_row, between_col] == player:
                # cannot jump over your own piece
                return False
            if self.board[between_row, between_col] == 0:
                # cannot jump over empty space
                return False

        return True


    def check_valid_move(self, row, col, direction, move_type):
        if self.board[row, col] != self.player:
            raise ValueError("Invalid move: Not your piece")
        next_row = row
        next_col = col
        between_row = row
        between_col = col
        # 0, 1, 2, 3 = up, right, down, left
        if move_type == "jump":
            if direction == 0:
                next_row = row - 2
                between_row = row - 1
            elif direction == 2:
                next_row = row + 2
                between_row = row + 1
            elif direction == 1:
                next_col = col + 2
                between_col = col + 1
            elif direction == 3:
                next_col = col - 2
                between_col = col - 1
        else:
            # Normal move
            if direction == 0:
                next_row = row - 1
            elif direction == 2:
                next_row = row + 1
            elif direction == 1:
                next_col = col + 1
            elif direction == 3:
                next_col = col - 1
        
        if next_row < 0 or next_row >= BOARD_DIM or next_col < 0 or next_col >= BOARD_DIM:
            raise ValueError("Invalid move: Out of bounds")
        
        if self.board[next_row, next_col] != 0:
            raise ValueError("Invalid move: Cell already occupied")
        
        if move_type == "jump":
            if self.board[between_row, between_col] == self.player:
                raise ValueError("Invalid move: Cannot jump over your own piece")
            if self.board[between_row, between_col] == 0:
                raise ValueError("Invalid move: Cannot jump over empty space")

        return next_row, next_col, between_row, between_col
    
    def render(self):
        '''
        Renders the board to make it a bit easier on the eyes.
        '''
        copy = self.board.copy()
        # map values to symbols
        symbols = {0: '.', 1: 'X', 2: 'O'}
        cols = copy.shape[1]

        # print column headers
        print("   " + " ".join(str(c) for c in range(cols)))

        # print each row
        for r, row in enumerate(copy):
            line = " ".join(symbols[val] for val in row)
            print(f"{r}  {line}")


    @staticmethod
    def to_info(action):
        info_map = {
            "row": None,
            "col": None,
            "move_type": None,
            "direction": None,
            "capture_row": None,
            "capture_col": None
        }
        if action < 25:
            info_map["move_type"] = "place"
            info_map["row"], info_map["col"] = divmod(action, 5)
        elif action < 125:
            action -= 25
            info_map["move_type"] = "move"
            index, info_map["direction"] = divmod(action, 4)
            info_map["row"], info_map["col"] = divmod(index, 5)
        else:
            action -= 125
            info_map["move_type"] = "jump"
            jump_index, capture_index = divmod(action, 25)
            info_map["capture_row"], info_map["capture_col"] = divmod(capture_index, 5)
            index, info_map["direction"] = divmod(jump_index, 4)
            info_map["row"], info_map["col"] = divmod(index, 5)
        return info_map

    
    @staticmethod
    def to_action(row, col, move_type, direction=None, capture_row = None, capture_col = None):
        if move_type == "place":
            return row * 5 + col
        if move_type == "move":
            return 25 + (row*5 + col) * 4 + direction
        if move_type == "jump":
            if capture_row is None or capture_col is None:
                raise ValueError("capture_row and capture_col must be provided for jump")
            jump_index = (row*5 + col) * 4 + direction
            capture_index = capture_row * 5 + capture_col
            return 125 + jump_index * 25 + capture_index

def move_tests():
    '''
    Tests the move function of the Choko_Env class.
    '''
    env = Choko_Env()
    env.freeze_turns = True # for debugging

    # testing that we can place a piece
    env.step(0)
    assert env.board[0, 0] == 1

    env.player = 1
    env.step(21)
    row, col = divmod(21, 5)
    assert env.board[row, col] == 1

    env.player = 1
    row = 3
    col = 2
    action = env.to_action(row=row, col=col, move_type="place")
    env.step(action)
    assert env.board[row, col] == 1

    # testing that we can move a piece
    row = 0
    col = 0
    direction = 2
    action = env.to_action(row=row, col=col, move_type="move", direction=direction)
    next_row, next_col, _, _ = env.check_valid_move(row, col, direction, move_type="move")

    env.step(action)
    assert env.board[row, col] == 0
    assert env.board[next_row, next_col] == 1

    row = 3
    col = 2
    direction = 1
    action = env.to_action(row=row, col=col, move_type="move", direction=direction)
    assert env.board[row, col] == 1
    next_row, next_col, _, _ = env.check_valid_move(row, col, direction, move_type="move")
    env.step(action)
    assert env.board[row, col] == 0
    assert env.board[next_row, next_col] == 1

    # testing that we can jump over a piece, no extra capture

    env.reset()
    env.freeze_turns = True
    first_row = 2
    first_col = 2
    action = env.to_action(row=first_row, col=first_col, move_type="place")
    env.step(action)
    assert env.board[first_row, first_col] == 1

    env.player = 2
    second_row = 2
    second_col = 3
    action = env.to_action(row=second_row, col=second_col, move_type="place")
    env.step(action)
    assert env.board[second_row, second_col] == 2

    action = env.to_action(
        row=first_row,
        col=first_col,
        move_type="jump",
        direction=1,
        capture_row=second_row,
        capture_col=second_col
    )

    env.player = 1
    next_row, next_col, between_row, between_col = env.check_valid_move(first_row, first_col, 1, move_type="jump")
    env.step(action)

    assert env.board[first_row, first_col] == 0
    assert env.board[next_row, next_col] == 1
    assert env.board[between_row, between_col] == 0

    # testing that we can jump over a piece, with extra capture

    env.reset()
    env.freeze_turns = True
    first_row = 2
    first_col = 2
    action = env.to_action(row=first_row, col=first_col, move_type="place")
    env.step(action)
    assert env.board[first_row, first_col] == 1

    env.player = 2
    second_row = 2
    second_col = 3
    action = env.to_action(row=second_row, col=second_col, move_type="place")
    env.step(action)
    assert env.board[second_row, second_col] == 2

    third_row = 0
    third_col = 0
    action = env.to_action(row=third_row, col=third_col, move_type="place")
    env.step(action)
    print(env.board)
    assert env.board[third_row, third_col] == 2

    env.player = 1
    action = env.to_action(
        row=first_row,
        col=first_col,
        move_type="jump",
        direction=1,
        capture_row=third_row,
        capture_col=third_col
    )
    next_row, next_col, between_row, between_col = env.check_valid_move(first_row, first_col, 1, move_type="jump")
    env.step(action)

    assert env.board[first_row, first_col] == 0
    assert env.board[next_row, next_col] == 1
    assert env.board[between_row, between_col] == 0
    assert env.board[third_row, third_col] == 0
    
def test_action_mask():
    '''
    Tests action masking and gameplay.
    '''
    env = Choko_Env()
    # for i in range(10000):
    #     obs, mask = env.reset()
    #     valid = np.flatnonzero(mask)
    #     action = np.random.choice(valid)
    #     assert action < 25
    
    # just two rounds
    obs, mask = env.reset()
    valid = np.flatnonzero(mask)
    action = np.random.choice(valid)
    assert action < 25
    state, reward, done, _ = env.step(action)
    obs, mask = state
    valid = np.flatnonzero(mask)
    action = np.random.choice(valid)
    state, reward, done, _ = env.step(action)

    # more rounds

    obs, mask = env.reset()
    valid = np.flatnonzero(mask)
    action = np.random.choice(valid)
    state, reward, done, _ = env.step(action)
    print("\n")
    print(env.board)
    print("\n")
    obs, mask = state
    while True:
        valid = np.flatnonzero(mask)
        action = np.random.choice(valid)
        state, reward, done, _ = env.step(action)
        if done:
            break
        obs, mask = state
        print("\n")
        print(env.board)
        print("\n")





if __name__ == "__main__":
    move_tests()
    print("\n\nAll move tests passed\n")

    test_action_mask()



