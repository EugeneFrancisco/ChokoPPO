import copy
import numpy as np

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------

WIN_SCORE: int = 10_000  # Large finite value for terminal wins


def clone_env(env):
    """Deep–copies the entire Choko_Env instance so the search can enjoy
    an isolated sandbox.  The environment is pure Python + NumPy, so the
    standard library’s ``copy.deepcopy`` is sufficient in practice.  If
    your environment ever acquires non‑picklable members (e.g. open file
    handles, CUDA tensors) swap this for a custom state‑dict copier.
    """
    return copy.deepcopy(env)


def legal_actions(env):
    """Returns a *list* of integer actions that are legal in the **current**
    Choko_Env position.
    """
    _, mask = env.fetch_obs_action_mask()
    return np.nonzero(mask)[0].tolist()


def evaluate_material(env, root_player: int) -> float:
    """Simple heuristic used at depth‑limit nodes: *material balance*.

    ``pieces_left`` counts yet‑to‑be‑dropped stones; board occupancy is
    counted via NumPy boolean sums.  Positive values are good for
    ``root_player``; negative values favour the opponent.
    """
    me = root_player
    opp = 3 - root_player

    def remaining(player):
        on_board = (env.board == player).sum()
        to_drop = env.pieces_left[player]
        return on_board + to_drop

    return float(remaining(me) - remaining(opp))


# ------------------------------------------------------------
# Minimax search compatible with *your* Choko_Env
# ------------------------------------------------------------

class MinimaxAgent:
    """Vanilla depth‑limited minimax with an optional heuristic.

    The implementation is intentionally lightweight so that it can slot
    straight into your PPO training loop (e.g. as an opponent during
    curriculum play).
    """

    def __init__(self, max_depth: int = 3, eval_fn=None):
        self.max_depth = max_depth
        self.eval_fn = eval_fn or evaluate_material

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def choose_action(self, env):
        """Return the *integer* action the search rates best for the
        *current* player found in ``env.player``.
        """
        root_player = env.player
        best_val = -float("inf")
        best_act = None

        for act in legal_actions(env):
            env_copy = clone_env(env)
            _, reward, _, _ = env_copy.step(act)  # advance one ply
            val = reward + self._minimax(env_copy, self.max_depth - 1,
                                         maximizing=False, root_player=root_player)
            if val > best_val:
                best_val, best_act = val, act
        return best_act
    

    # ------------------------------------------------------------------
    # Internal recursive minimax
    # ------------------------------------------------------------------
    def _minimax(self, env, depth: int, *, maximizing: bool, root_player: int):
        # --- Terminal or depth‑limit -------------------------------------------------
        status = env.evaluate_termination()  # "won", "draw", or "ongoing"
        if status != "ongoing" or depth == 0:
            if status == "won":
                # ``env.player`` has *already* switched after the last move, so
                # the *previous* player (3‑env.player) made the winning move.
                winner = 3 - env.player
                return (WIN_SCORE if winner == root_player else -WIN_SCORE)
            elif status == "draw":
                return 0.0
            else:  # Depth limit
                return self.eval_fn(env, root_player)

        # --- Non‑terminal: expand children -----------------------------------------
        acts = legal_actions(env)
        if not acts:
            # No legal moves -> stalemate = draw in this formulation
            return 0.0

        if maximizing:
            value = -float("inf")
            for act in acts:
                child = clone_env(env)
                _, reward, _, _ = child.step(act)
                value = max(value, reward + self._minimax(child, depth - 1,
                                                           maximizing=False,
                                                           root_player=root_player))
            return value
        else:
            value = float("inf")
            for act in acts:
                child = clone_env(env)
                _, reward, _, _ = child.step(act)
                value = min(value, reward + self._minimax(child, depth - 1,
                                                           maximizing=True,
                                                           root_player=root_player))
            return value

if __name__ == "__main__":
    from envs.choko_env import Choko_Env  # Adjust this import based on your project structure

    env = Choko_Env()
    agent = MinimaxAgent(max_depth=3)

    # Example usage: reset the environment and choose an action
    raw_obs, mask = env.reset()
    action = agent.choose_action(env)
    print(f"Chosen action: {action}")