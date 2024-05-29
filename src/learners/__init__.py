from .q_learner import QLearner
from .nq_learner import NQLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .ppo_learner import PPOLearner
# from .happo_learner import HAPPO

REGISTRY = {}

# REGISTRY["q_learner"] = QLearner
REGISTRY["nq_learner"] = NQLearner
# REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["ppo_learner"] = PPOLearner
# REGISTRY["happo_learner"] = HAPPO