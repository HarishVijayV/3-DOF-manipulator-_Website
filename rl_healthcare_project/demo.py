#!/usr/bin/env python3
"""
Quick demonstration of the RL Healthcare Project.
This script provides a simple way to test the project setup and see basic functionality.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.diabetes_env import DiabetesEnv
from algorithms.dqn.dqn_agent import DQNAgent


def demo_environment():
    """Demonstrate the diabetes environment with random actions."""
    print("🏥 DIABETES ENVIRONMENT DEMO")
    print("=" * 50)
    
    # Create environment
    env = DiabetesEnv()
    
    # Reset environment
    state = env.reset()
    print(f"Initial state: {state}")
    env.render()
    
    # Run a few steps with random actions
    total_reward = 0
    glucose_history = []
    actions_taken = []
    
    for step in range(10):
        action = env.action_space.sample()  # Random action
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        glucose_history.append(info['glucose_level'])
        actions_taken.append(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action} (Insulin: {env.insulin_doses[action]:.1f} units)")
        print(f"  Glucose: {info['glucose_level']:.1f} mg/dL")
        print(f"  Reward: {reward:.2f}")
        print(f"  In range: {info['in_target_range']}")
        
        if done:
            break
    
    print(f"\nDemo completed!")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average glucose: {np.mean(glucose_history):.1f} mg/dL")
    
    return glucose_history, actions_taken


def demo_agent_training():
    """Demonstrate a quick training session with the DQN agent."""
    print("\n🤖 DQN AGENT TRAINING DEMO")
    print("=" * 50)
    
    # Create environment
    env = DiabetesEnv()
    
    # Create agent
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        lr=0.001,
        memory_size=1000,
        batch_size=16
    )
    
    print("Training agent for 50 episodes...")
    
    episode_rewards = []
    
    for episode in range(50):
        state = env.reset()
        episode_reward = 0
        
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train the agent
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    print("Training demo completed!")
    
    # Test the trained agent
    print("\nTesting trained agent...")
    state = env.reset()
    total_reward = 0
    glucose_levels = []
    actions = []
    
    done = False
    while not done:
        action = agent.act(state, training=False)  # No exploration
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        glucose_levels.append(info['glucose_level'])
        actions.append(action)
        state = next_state
    
    time_in_range = sum(1 for g in glucose_levels if 80 <= g <= 120) / len(glucose_levels) * 100
    
    print(f"Test results:")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Time in range: {time_in_range:.1f}%")
    print(f"  Average glucose: {np.mean(glucose_levels):.1f} mg/dL")
    
    return episode_rewards, glucose_levels, actions


def create_demo_plots(glucose_history, episode_rewards):
    """Create demonstration plots."""
    print("\n📊 CREATING DEMO PLOTS")
    print("=" * 50)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Glucose levels over time
    axes[0, 0].plot(glucose_history, 'b-', linewidth=2, label='Glucose Level')
    axes[0, 0].axhline(y=80, color='g', linestyle='--', alpha=0.7, label='Target Min')
    axes[0, 0].axhline(y=120, color='g', linestyle='--', alpha=0.7, label='Target Max')
    axes[0, 0].fill_between(range(len(glucose_history)), 80, 120, alpha=0.2, color='green')
    axes[0, 0].set_title('Blood Glucose Levels')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Glucose (mg/dL)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Training rewards
    axes[0, 1].plot(episode_rewards, 'r-', linewidth=2)
    axes[0, 1].set_title('Training Progress')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Episode Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Glucose distribution
    axes[1, 0].hist(glucose_history, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(x=80, color='g', linestyle='--', linewidth=2, label='Target Range')
    axes[1, 0].axvline(x=120, color='g', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Glucose Distribution')
    axes[1, 0].set_xlabel('Glucose (mg/dL)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Reward moving average
    if len(episode_rewards) >= 10:
        window = min(10, len(episode_rewards))
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        axes[1, 1].plot(range(window-1, len(episode_rewards)), moving_avg, 'purple', linewidth=2)
        axes[1, 1].set_title('Moving Average Reward')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("Demo plots created successfully!")


def main():
    """Run the complete demonstration."""
    print("🚀 RL HEALTHCARE PROJECT DEMONSTRATION")
    print("=" * 60)
    print("This demo will show you the basic functionality of the project.")
    print("It includes environment testing and a quick agent training session.")
    print("=" * 60)
    
    try:
        # Demo 1: Environment
        glucose_history, actions = demo_environment()
        
        # Demo 2: Agent Training
        episode_rewards, test_glucose, test_actions = demo_agent_training()
        
        # Demo 3: Plots
        create_demo_plots(test_glucose, episode_rewards)
        
        print("\n✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("🎯 Next steps:")
        print("1. Run full training: python train_diabetes_dqn.py --episodes 1000")
        print("2. Explore the Jupyter notebooks in the notebooks/ directory")
        print("3. Implement additional environments (hypertension, depression)")
        print("4. Extend to PPO and A3C algorithms")
        print("5. Build the web application component")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print("Please check your installation and try again.")
        print("Make sure all dependencies are installed: pip install -r requirements.txt")


if __name__ == "__main__":
    main()