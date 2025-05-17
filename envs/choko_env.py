from gym import spaces
import numpy as np

NUM_ACTIONS: int = (25) + (25 * 4) + (25 * 4 * 25) # 2625 possible actions
BOARD_DIM: int = 5
NUM_PIECES_PER_PLAYER: int = 12

class Choko_Env:
    def __init__(self):
        self.obs_space = spaces.Box(low = 0, high = 2, shape = (BOARD_DIM, BOARD_DIM), dtype = np.int8)
        self.board = np.zeros((BOARD_DIM, BOARD_DIM), dtype = np.int8)
        self.player = 1
        self.drop_initiative = 1
        self.action_space = spaces.Discrete(NUM_ACTIONS)
    
    def reset(self):
        self.board = np.zeros((5, 5), dtype = np.int8)
        self.player = 1
        self.drop_initiative = 1
        return self.board
    
    def step(self, action):
        # TODO change drop initiative
        assert self.action_space.contains(action), f"Invalid action {action}"

        # First 25 moves are for placing a piece
        if action < 25:
            row, col = divmod(action, 5)
            if self.board[row, col] == 0:
                self.board[row, col] = self.player
            else:
                raise ValueError("Invalid move: Cell already occupied")
        
        # next 48 moves are for moving a piece
        elif action < 125:
            # TODO implement board sum
            if self.drop_initiative != self.player:
                raise ValueError("Invalid move: Not your drop initiative")
            action -= 25
            index, direction = divmod(action, 4)
            row, col = divmod(index, 5)
            
            next_row, next_col, _, _ = self.check_valid_move(row, col, direction)
            self.board[next_row, next_col] = self.player
            self.board[row, col] = 0
        
        else:
            # TODO, implement board sum
            action -= 125
            jump_index, capture_index = divmod(action, 25)
            capture_row, capture_col = divmod(capture_index, 5)
            og_index, direction = divmod(jump_index, 4)
            og_row, og_col = divmod(og_index, 5)

            next_row, next_col, between_row, between_col = self.check_valid_move(og_row, og_col, direction, jump=True)

            # perform the jump
            self.board[next_row, next_col] = self.player
            self.board[og_row, og_col] = 0

            # first capture:
            self.board[between_row, between_col] = 0

            # extra capture
            if self.board[capture_row, capture_col] == self.player:
                raise ValueError("Invalid move: Cannot capture your own piece")
            
            self.board[capture_row, capture_col] = 0
        
        # TODO delete this after debugging
        # self.player = 3 - self.player
        # TODO: return the flattened observations.
            
    def board_sum(self):
        return self.board[self.board == self.player].sum()
    
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

    def check_valid_move(self, row, col, direction, jump=False):
        if self.board[row, col] != self.player:
            raise ValueError("Invalid move: Not your piece")
        next_row = row
        next_col = col
        between_row = row
        between_col = col
        # 0, 1, 2, 3 = up, right, down, left
        if jump:
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
            if self.board[between_row, between_col] == self.player:
                raise ValueError("Invalid move: Cannot jump over your own piece")
            if self.board[between_row, between_col] == 0:
                raise ValueError("Invalid move: Cannot jump over empty space")
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

        return next_row, next_col, between_row, between_col


def run_tests():
    env = Choko_Env()

    # testing that we can place a piece
    env.step(0)
    assert env.board[0, 0] == 1

    env.step(21)
    row, col = divmod(21, 5)
    assert env.board[row, col] == 1

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
    next_row, next_col, _, _ = env.check_valid_move(row, col, direction)

    env.step(action)
    assert env.board[row, col] == 0
    assert env.board[next_row, next_col] == 1

    row = 3
    col = 2
    direction = 1
    action = env.to_action(row=row, col=col, move_type="move", direction=direction)
    assert env.board[row, col] == 1
    next_row, next_col, _, _ = env.check_valid_move(row, col, direction)
    env.step(action)
    assert env.board[row, col] == 0
    assert env.board[next_row, next_col] == 1

    # testing that we can jump over a piece, no extra capture

    env.reset()
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
    next_row, next_col, between_row, between_col = env.check_valid_move(first_row, first_col, 1, jump=True)
    env.step(action)

    assert env.board[first_row, first_col] == 0
    assert env.board[next_row, next_col] == 1
    assert env.board[between_row, between_col] == 0

    # testing that we can jump over a piece, with extra capture

    env.reset()
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
    next_row, next_col, between_row, between_col = env.check_valid_move(first_row, first_col, 1, jump=True)
    env.step(action)

    assert env.board[first_row, first_col] == 0
    assert env.board[next_row, next_col] == 1
    assert env.board[between_row, between_col] == 0

if __name__ == "__main__":
    run_tests()



