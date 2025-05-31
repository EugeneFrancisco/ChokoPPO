# config.py

# training loop hyperparameters
BATCH_SIZE     = 128
ROLLOUT_LENGTH = 4096
HIDDEN_DIM     = 64
NUM_EPOCHS     = 6
NUM_ITERATIONS = 5000
LEARNING_RATE = 1e-4

# replay buffer hyperparameters
MAX_BUFFER_SIZE = 4096
NUM_TASKS = 2

# environment / action settings
NUM_ACTIONS = (25) + (25 * 4) + (25 * 4 * 25)  # 2625 possible actions
EVAL_ROUNDS = 100
GAMMA = 0.99

# PPO-specific coefficients
C1      = 0.5
C2      = 0.01
EPS_CLIP = 0.2
NEG_INF  = -1e10
LAM = 0.95

# Q leanring hyperparameters
EPSILON = 0.1
TD_GAP = 5
REPLAY_BUFFER_SIZE = 20000
TARGET_UPDATE_FREQ = 100
NUM_GRADIENT_STEPS_PER_UPDATE = 1
NUM_Q_LEARNING_ITERATIONS = 200000
LOGGING_FREQ = 5
EVAL_FREQ = 200
FROZEN_UPDATE_FREQ = 1000
Q_LEARNING_RATE = 1e-3
