from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn.functional as F

# Imports everything from agent2 to avoid rewriting the replay buffer
from .agent2 import DQNAgent as DQNAgent2, Transition, ReplayBuffer, DEVICE

class DQNAgent3(DQNAgent2):
    """Task 3 Agent - Inherits from Task 2 but adds proper Boltzmann sampling."""
    
    def select_action_boltzmann(self, state: np.ndarray, temperature: float) -> int:
        """Task 3.2: Boltzmann Exploration using Softmax."""
        # Convert state to tensor
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            # Get raw Q-values for the 2 actions
            q_values = self.online(state_t)[0] 
            
        # Scale Q-values by the temperature parameter (prevent division by zero)
        scaled_q_values = q_values / max(temperature, 1e-3)
        
        # Apply Softmax to compute action probabilities
        probabilities = torch.softmax(scaled_q_values, dim=0)
        
        # Sample an action based on the calculated distribution
        m = torch.distributions.Categorical(probabilities)
        action = m.sample().item()
        
        return action