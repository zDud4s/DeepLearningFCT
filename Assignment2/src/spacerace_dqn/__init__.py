"""SpaceRace DQN — Task 1 reference implementation"""
from .agent import DEVICE, BasicDQNAgent
from .config import Config
from .env import SpaceRaceEnv, make_env, reset_env
from .evaluator import Evaluator
from .live_player import LivePlayer
from .network import SmallQNetwork
from .plotting import plot_baselines, plot_training
from .policies import (
    AlwaysUpPolicy,
    DQNInferencePolicy,
    Policy,
    RandomPolicy,
    RGBHeuristicPolicy,
    SemanticHeuristicPolicy,
    policy_factory,
)
from .preprocessing import FrameStacker, preprocess_obs
from .recording import GifRecorder
from .submission import SubmissionPackager
from .trainer import DQNTrainer, epsilon_by_step, safe_rate, set_seed

__all__ = [
    "BasicDQNAgent",
    "Config",
    "DEVICE",
    "DQNInferencePolicy",
    "DQNTrainer",
    "Evaluator",
    "FrameStacker",
    "GifRecorder",
    "LivePlayer",
    "Policy",
    "AlwaysUpPolicy",
    "RandomPolicy",
    "RGBHeuristicPolicy",
    "SemanticHeuristicPolicy",
    "SmallQNetwork",
    "SpaceRaceEnv",
    "SubmissionPackager",
    "epsilon_by_step",
    "make_env",
    "plot_baselines",
    "plot_training",
    "policy_factory",
    "preprocess_obs",
    "reset_env",
    "safe_rate",
    "set_seed",
]
