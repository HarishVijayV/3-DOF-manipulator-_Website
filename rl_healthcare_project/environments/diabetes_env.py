import gym
import numpy as np
from gym import spaces
from typing import Tuple, Dict, Any


class DiabetesEnv(gym.Env):
    """
    A gym environment for simulating diabetes management through insulin dosage optimization.
    
    State: [blood_glucose, time_since_meal, carb_intake, previous_insulin, exercise_level]
    Action: insulin_dosage (discrete: 0=low, 1=medium, 2=high)
    Reward: Penalty for being outside target glucose range, bonus for staying in range
    """
    
    def __init__(self, patient_params: Dict[str, float] = None):
        super(DiabetesEnv, self).__init__()
        
        # Patient-specific parameters
        if patient_params is None:
            patient_params = {
                'insulin_sensitivity': 1.0,
                'carb_ratio': 15.0,  # grams of carbs per unit of insulin
                'basal_glucose': 100.0,  # baseline glucose level
                'glucose_variability': 10.0  # natural variation
            }
        self.patient_params = patient_params
        
        # Action space: 3 discrete insulin levels (low, medium, high)
        self.action_space = spaces.Discrete(3)
        
        # State space: [glucose, time_since_meal, carb_intake, prev_insulin, exercise]
        self.observation_space = spaces.Box(
            low=np.array([50.0, 0.0, 0.0, 0.0, 0.0]),
            high=np.array([400.0, 8.0, 100.0, 20.0, 2.0]),
            dtype=np.float32
        )
        
        # Target glucose range (mg/dL)
        self.target_min = 80.0
        self.target_max = 120.0
        
        # Current state
        self.state = None
        self.steps = 0
        self.max_steps = 288  # 24 hours * 12 (5-min intervals)
        
        # Action mapping
        self.insulin_doses = [2.0, 5.0, 8.0]  # Low, medium, high insulin units
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.steps = 0
        # Initialize with random but realistic state
        self.state = np.array([
            np.random.uniform(90, 110),  # glucose level
            np.random.uniform(0, 4),     # time since meal (hours)
            np.random.uniform(0, 50),    # carb intake (grams)
            0.0,                         # previous insulin
            np.random.uniform(0, 1)      # exercise level
        ], dtype=np.float32)
        
        return self.state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        if self.state is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        
        # Get insulin dose from action
        insulin_dose = self.insulin_doses[action]
        
        # Current state values
        glucose, time_since_meal, carb_intake, prev_insulin, exercise = self.state
        
        # Simulate glucose dynamics
        new_glucose = self._simulate_glucose_change(
            glucose, insulin_dose, carb_intake, exercise
        )
        
        # Calculate reward
        reward = self._calculate_reward(new_glucose, insulin_dose)
        
        # Update state for next step
        self.state = np.array([
            new_glucose,
            np.random.uniform(0, 4),  # New meal timing
            np.random.uniform(0, 50), # New carb intake
            insulin_dose,
            np.random.uniform(0, 1)   # New exercise level
        ], dtype=np.float32)
        
        self.steps += 1
        done = self.steps >= self.max_steps
        
        info = {
            'glucose_level': new_glucose,
            'in_target_range': self.target_min <= new_glucose <= self.target_max,
            'insulin_given': insulin_dose,
            'hypoglycemic': new_glucose < 70,
            'hyperglycemic': new_glucose > 180
        }
        
        return self.state, reward, done, info
    
    def _simulate_glucose_change(self, current_glucose: float, insulin: float, 
                               carbs: float, exercise: float) -> float:
        """Simulate blood glucose level change based on various factors."""
        
        # Base glucose change (tends toward basal level)
        glucose_change = (self.patient_params['basal_glucose'] - current_glucose) * 0.1
        
        # Effect of carbohydrates (increases glucose)
        carb_effect = carbs * 3.0  # Simplified carb absorption
        
        # Effect of insulin (decreases glucose)
        insulin_effect = -insulin * self.patient_params['insulin_sensitivity'] * 15.0
        
        # Effect of exercise (decreases glucose)
        exercise_effect = -exercise * 20.0
        
        # Add some random variation
        noise = np.random.normal(0, self.patient_params['glucose_variability'])
        
        # Calculate new glucose level
        new_glucose = current_glucose + glucose_change + carb_effect + insulin_effect + exercise_effect + noise
        
        # Ensure glucose stays within physiological bounds
        new_glucose = np.clip(new_glucose, 40.0, 500.0)
        
        return new_glucose
    
    def _calculate_reward(self, glucose: float, insulin: float) -> float:
        """Calculate reward based on glucose level and insulin usage."""
        
        # Base reward for staying in target range
        if self.target_min <= glucose <= self.target_max:
            reward = 10.0
        else:
            # Penalty increases quadratically with distance from target range
            if glucose < self.target_min:
                distance = self.target_min - glucose
                reward = -0.1 * (distance ** 2)
            else:  # glucose > self.target_max
                distance = glucose - self.target_max
                reward = -0.1 * (distance ** 2)
        
        # Severe penalty for dangerous levels
        if glucose < 70:  # Hypoglycemia
            reward -= 50.0
        elif glucose > 250:  # Severe hyperglycemia
            reward -= 30.0
        
        # Small penalty for insulin usage (encourage minimal effective dose)
        reward -= insulin * 0.1
        
        return reward
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment state."""
        if self.state is None:
            print("Environment not initialized")
            return
        
        glucose, time_since_meal, carb_intake, prev_insulin, exercise = self.state
        
        print(f"Step: {self.steps}")
        print(f"Blood Glucose: {glucose:.1f} mg/dL")
        print(f"Time since meal: {time_since_meal:.1f} hours")
        print(f"Carb intake: {carb_intake:.1f} grams")
        print(f"Previous insulin: {prev_insulin:.1f} units")
        print(f"Exercise level: {exercise:.2f}")
        print(f"In target range: {self.target_min <= glucose <= self.target_max}")
        print("-" * 40)
    
    def get_patient_info(self) -> Dict[str, float]:
        """Return patient-specific parameters."""
        return self.patient_params.copy()


if __name__ == "__main__":
    # Test the environment
    env = DiabetesEnv()
    
    print("Testing Diabetes Environment")
    print("=" * 50)
    
    obs = env.reset()
    print("Initial state:", obs)
    env.render()
    
    # Run a few steps
    for i in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)
        
        print(f"\nAction: {action} (Insulin: {env.insulin_doses[action]} units)")
        print(f"Reward: {reward:.2f}")
        print(f"Info: {info}")
        env.render()
        
        if done:
            break