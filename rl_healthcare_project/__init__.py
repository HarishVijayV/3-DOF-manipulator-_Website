"""
RL Healthcare Project - AI-Driven Drug Dosage Optimization using Reinforcement Learning

This package implements multiple reinforcement learning algorithms to optimize drug dosage 
for treating diabetes, hypertension, and depression.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .environments.diabetes_env import DiabetesEnv
from .algorithms.dqn.dqn_agent import DQNAgent

__all__ = ["DiabetesEnv", "DQNAgent"]