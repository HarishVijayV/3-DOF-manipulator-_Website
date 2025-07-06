#!/usr/bin/env python3
"""
Training script for DQN agent on diabetes management environment.
This script demonstrates the complete pipeline from environment setup to model training and evaluation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.diabetes_env import DiabetesEnv
from algorithms.dqn.dqn_agent import DQNAgent


def create_patient_population(n_patients: int = 5):
    """Create a diverse population of patients with different characteristics."""
    patients = []
    
    for i in range(n_patients):
        # Generate diverse patient parameters
        patient_params = {
            'insulin_sensitivity': np.random.uniform(0.5, 1.5),
            'carb_ratio': np.random.uniform(10, 20),
            'basal_glucose': np.random.uniform(90, 110),
            'glucose_variability': np.random.uniform(5, 15)
        }
        patients.append(patient_params)
    
    return patients


def evaluate_agent(agent: DQNAgent, env: DiabetesEnv, n_episodes: int = 10):
    """Evaluate the trained agent on the environment."""
    total_rewards = []
    glucose_control_metrics = []
    hypoglycemic_events = []
    hyperglycemic_events = []
    
    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        glucose_levels = []
        hypo_events = 0
        hyper_events = 0
        in_range_steps = 0
        
        done = False
        while not done:
            action = agent.act(state, training=False)  # No exploration during evaluation
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            glucose_levels.append(info['glucose_level'])
            
            if info['hypoglycemic']:
                hypo_events += 1
            if info['hyperglycemic']:
                hyper_events += 1
            if info['in_target_range']:
                in_range_steps += 1
            
            state = next_state
        
        total_rewards.append(episode_reward)
        hypoglycemic_events.append(hypo_events)
        hyperglycemic_events.append(hyper_events)
        
        # Calculate time in range percentage
        time_in_range = (in_range_steps / env.max_steps) * 100
        glucose_control_metrics.append({
            'time_in_range': time_in_range,
            'mean_glucose': np.mean(glucose_levels),
            'glucose_std': np.std(glucose_levels),
            'min_glucose': np.min(glucose_levels),
            'max_glucose': np.max(glucose_levels)
        })
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_time_in_range': np.mean([m['time_in_range'] for m in glucose_control_metrics]),
        'mean_hypoglycemic_events': np.mean(hypoglycemic_events),
        'mean_hyperglycemic_events': np.mean(hyperglycemic_events),
        'glucose_metrics': glucose_control_metrics
    }


def train_dqn_agent(config: dict):
    """Main training function for DQN agent."""
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results/diabetes_dqn_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Starting DQN training for diabetes management")
    print(f"Results will be saved to: {results_dir}")
    print("=" * 60)
    
    # Create patient population
    patients = create_patient_population(config['n_patients'])
    print(f"Created {len(patients)} diverse patients")
    
    # Initialize environment with first patient
    env = DiabetesEnv(patients[0])
    
    # Initialize DQN agent
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        lr=config['learning_rate'],
        gamma=config['gamma'],
        epsilon=config['epsilon_start'],
        epsilon_min=config['epsilon_min'],
        epsilon_decay=config['epsilon_decay'],
        memory_size=config['memory_size'],
        batch_size=config['batch_size'],
        target_update=config['target_update']
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    evaluation_results = []
    
    print(f"Training for {config['n_episodes']} episodes...")
    
    for episode in tqdm(range(config['n_episodes']), desc="Training Episodes"):
        
        # Rotate through patients for diversity
        patient_idx = episode % len(patients)
        env = DiabetesEnv(patients[patient_idx])
        
        state = env.reset()
        episode_reward = 0
        steps = 0
        
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train the agent
            loss = agent.replay()
            
            episode_reward += reward
            steps += 1
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        agent.rewards_history.append(episode_reward)
        
        # Periodic evaluation
        if (episode + 1) % config['eval_frequency'] == 0:
            print(f"\nEpisode {episode + 1}")
            print(f"Episode Reward: {episode_reward:.2f}")
            print(f"Average Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            
            # Evaluate on multiple patients
            eval_results = {}
            for i, patient in enumerate(patients):
                eval_env = DiabetesEnv(patient)
                patient_eval = evaluate_agent(agent, eval_env, n_episodes=5)
                eval_results[f'patient_{i}'] = patient_eval
            
            evaluation_results.append({
                'episode': episode + 1,
                'results': eval_results
            })
            
            # Print evaluation summary
            avg_time_in_range = np.mean([eval_results[f'patient_{i}']['mean_time_in_range'] 
                                       for i in range(len(patients))])
            print(f"Average Time in Range: {avg_time_in_range:.1f}%")
    
    print("\nTraining completed!")
    
    # Save the trained model
    model_path = os.path.join(results_dir, "dqn_diabetes_model.pth")
    agent.save(model_path)
    
    # Final comprehensive evaluation
    print("\nRunning final evaluation...")
    final_evaluation = {}
    for i, patient in enumerate(patients):
        eval_env = DiabetesEnv(patient)
        patient_eval = evaluate_agent(agent, eval_env, n_episodes=20)
        final_evaluation[f'patient_{i}'] = patient_eval
        
        print(f"Patient {i+1} - Time in Range: {patient_eval['mean_time_in_range']:.1f}%, "
              f"Avg Reward: {patient_eval['mean_reward']:.2f}")
    
    # Save results
    results_summary = {
        'config': config,
        'episode_rewards': episode_rewards,
        'evaluation_results': evaluation_results,
        'final_evaluation': final_evaluation,
        'training_stats': agent.get_training_stats()
    }
    
    # Save results as CSV for analysis
    results_df = pd.DataFrame({
        'episode': range(1, len(episode_rewards) + 1),
        'reward': episode_rewards,
        'episode_length': episode_lengths
    })
    results_df.to_csv(os.path.join(results_dir, 'training_results.csv'), index=False)
    
    # Plot training progress
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    window = 100
    if len(episode_rewards) >= window:
        moving_avg = pd.Series(episode_rewards).rolling(window).mean()
        plt.plot(moving_avg)
        plt.title(f'Moving Average Reward (window={window})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(episode_lengths)
    plt.title('Episode Lengths')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    if agent.losses:
        plt.plot(agent.losses)
        plt.title('Training Loss')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True)
    
    plt.subplot(2, 3, 5)
    # Time in range over evaluations
    if evaluation_results:
        eval_episodes = [er['episode'] for er in evaluation_results]
        avg_tir = []
        for er in evaluation_results:
            tir_values = [er['results'][f'patient_{i}']['mean_time_in_range'] 
                         for i in range(len(patients))]
            avg_tir.append(np.mean(tir_values))
        
        plt.plot(eval_episodes, avg_tir, 'o-')
        plt.title('Time in Range (Evaluation)')
        plt.xlabel('Episode')
        plt.ylabel('Time in Range (%)')
        plt.grid(True)
    
    plt.subplot(2, 3, 6)
    # Final evaluation summary
    patient_rewards = [final_evaluation[f'patient_{i}']['mean_reward'] 
                      for i in range(len(patients))]
    patient_tir = [final_evaluation[f'patient_{i}']['mean_time_in_range'] 
                  for i in range(len(patients))]
    
    plt.scatter(patient_rewards, patient_tir)
    plt.xlabel('Mean Reward')
    plt.ylabel('Time in Range (%)')
    plt.title('Final Performance per Patient')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'training_plots.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nResults saved to: {results_dir}")
    print(f"Model saved to: {model_path}")
    
    return agent, results_summary


def main():
    parser = argparse.ArgumentParser(description='Train DQN agent for diabetes management')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--memory_size', type=int, default=10000, help='Replay buffer size')
    parser.add_argument('--eval_freq', type=int, default=100, help='Evaluation frequency')
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'n_episodes': args.episodes,
        'n_patients': 5,
        'learning_rate': args.lr,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': args.memory_size,
        'batch_size': args.batch_size,
        'target_update': 100,
        'eval_frequency': args.eval_freq
    }
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Train the agent
    agent, results = train_dqn_agent(config)
    
    print("Training completed successfully!")
    return agent, results


if __name__ == "__main__":
    main()