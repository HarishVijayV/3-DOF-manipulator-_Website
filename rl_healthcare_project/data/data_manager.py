#!/usr/bin/env python3
"""
Data Manager for RL Healthcare Project

This module provides a unified interface for managing patient data, clinical guidelines,
and historical treatment records for training RL agents.
"""

import os
import sys
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.generate_patient_data import PatientDataGenerator
from data.clinical_guidelines import ClinicalGuidelines


class HealthcareDataManager:
    """Unified data manager for RL healthcare project."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.patient_generator = PatientDataGenerator()
        self.clinical_guidelines = ClinicalGuidelines()
        
        # Data paths
        self.patient_data_dir = os.path.join(data_dir, "synthetic_patients")
        self.guidelines_dir = os.path.join(data_dir, "clinical_guidelines")
        self.validation_dir = os.path.join(data_dir, "validation_datasets")
        
        # Initialize directories
        os.makedirs(self.patient_data_dir, exist_ok=True)
        os.makedirs(self.guidelines_dir, exist_ok=True)
        os.makedirs(self.validation_dir, exist_ok=True)
        
        # Loaded data cache
        self._patients_cache = {}
        self._historical_cache = {}
    
    def setup_datasets(self, regenerate: bool = False):
        """Setup all required datasets for RL training."""
        print("🏥 SETTING UP RL HEALTHCARE DATASETS")
        print("=" * 50)
        
        # Check if data already exists
        train_file = os.path.join(self.patient_data_dir, "train_patients.json")
        
        if os.path.exists(train_file) and not regenerate:
            print("✅ Patient data already exists. Use regenerate=True to recreate.")
        else:
            print("📊 Generating patient populations...")
            self._generate_patient_datasets()
        
        # Always regenerate guidelines (they're quick)
        print("📋 Generating clinical guidelines...")
        self._generate_clinical_guidelines()
        
        # Generate validation datasets
        print("🧪 Creating validation scenarios...")
        self._create_validation_datasets()
        
        print("✅ All datasets ready!")
    
    def _generate_patient_datasets(self):
        """Generate comprehensive patient datasets."""
        
        # Training set - large and diverse
        train_patients = self.patient_generator.generate_patient_population(n_patients=500)
        self.patient_generator.save_patient_data(
            train_patients, 
            os.path.join(self.patient_data_dir, "train_patients.json")
        )
        
        # Validation set - smaller, for hyperparameter tuning
        val_patients = self.patient_generator.generate_patient_population(n_patients=100)
        self.patient_generator.save_patient_data(
            val_patients,
            os.path.join(self.patient_data_dir, "val_patients.json")
        )
        
        # Test set - challenging cases
        test_patients = self.patient_generator.generate_patient_population(n_patients=50)
        self.patient_generator.save_patient_data(
            test_patients,
            os.path.join(self.patient_data_dir, "test_patients.json")
        )
        
        # Generate historical data for training patients
        print("Generating historical treatment records...")
        self._generate_historical_datasets(train_patients[:50])  # First 50 patients
        
        # Create patient summaries
        train_summary = self.patient_generator.create_patient_summary(train_patients)
        train_summary.to_csv(
            os.path.join(self.patient_data_dir, "train_summary.csv"), 
            index=False
        )
        
        print(f"Generated {len(train_patients)} training patients")
        print(f"Generated {len(val_patients)} validation patients")
        print(f"Generated {len(test_patients)} test patients")
    
    def _generate_historical_datasets(self, patients: List[Dict]):
        """Generate historical treatment data for patients."""
        all_historical = []
        
        for i, patient in enumerate(patients):
            # Generate 3 months of historical data
            historical_df = self.patient_generator.generate_historical_data(
                patient, days=90
            )
            all_historical.append(historical_df)
            
            # Save individual patient histories for detailed analysis
            if i < 10:  # First 10 patients
                historical_df.to_csv(
                    os.path.join(
                        self.patient_data_dir, 
                        f"history_{patient['patient_id']}.csv"
                    ),
                    index=False
                )
        
        # Combine all historical data
        combined_historical = pd.concat(all_historical, ignore_index=True)
        combined_historical.to_csv(
            os.path.join(self.patient_data_dir, "combined_historical.csv"),
            index=False
        )
        
        print(f"Generated {len(combined_historical)} historical records")
    
    def _generate_clinical_guidelines(self):
        """Generate and save clinical guidelines."""
        self.clinical_guidelines.save_guidelines(
            os.path.join(self.guidelines_dir, "diabetes_guidelines.json")
        )
        
        # Save validation scenarios
        from data.clinical_guidelines import create_validation_scenarios
        scenarios = create_validation_scenarios()
        with open(os.path.join(self.guidelines_dir, "validation_scenarios.json"), 'w') as f:
            json.dump(scenarios, f, indent=2)
    
    def _create_validation_datasets(self):
        """Create specific validation datasets for testing RL agents."""
        
        # Edge case patients
        edge_cases = self._create_edge_case_patients()
        with open(os.path.join(self.validation_dir, "edge_cases.json"), 'w') as f:
            json.dump(edge_cases, f, indent=2)
        
        # Benchmark scenarios
        benchmarks = self._create_benchmark_scenarios()
        with open(os.path.join(self.validation_dir, "benchmarks.json"), 'w') as f:
            json.dump(benchmarks, f, indent=2)
        
        print(f"Created {len(edge_cases)} edge case patients")
        print(f"Created {len(benchmarks)} benchmark scenarios")
    
    def _create_edge_case_patients(self) -> List[Dict]:
        """Create challenging edge case patients for testing."""
        edge_cases = []
        
        # Extremely insulin sensitive patient
        edge_cases.append({
            'patient_id': 'edge_hypersensitive',
            'description': 'Extremely insulin sensitive - prone to hypoglycemia',
            'params': {
                'insulin_sensitivity': 3.0,
                'carb_ratio': 5.0,
                'basal_glucose': 80.0,
                'glucose_variability': 30.0
            }
        })
        
        # Extremely insulin resistant patient
        edge_cases.append({
            'patient_id': 'edge_resistant',
            'description': 'Extremely insulin resistant - difficult to control',
            'params': {
                'insulin_sensitivity': 0.1,
                'carb_ratio': 50.0,
                'basal_glucose': 200.0,
                'glucose_variability': 50.0
            }
        })
        
        # Highly variable patient
        edge_cases.append({
            'patient_id': 'edge_variable',
            'description': 'Highly unpredictable glucose patterns',
            'params': {
                'insulin_sensitivity': 1.0,
                'carb_ratio': 15.0,
                'basal_glucose': 120.0,
                'glucose_variability': 60.0
            }
        })
        
        # Dawn phenomenon patient
        edge_cases.append({
            'patient_id': 'edge_dawn',
            'description': 'Strong dawn phenomenon - morning glucose spikes',
            'params': {
                'insulin_sensitivity': 0.8,
                'carb_ratio': 12.0,
                'basal_glucose': 140.0,
                'glucose_variability': 25.0,
                'dawn_effect': True
            }
        })
        
        return edge_cases
    
    def _create_benchmark_scenarios(self) -> List[Dict]:
        """Create benchmark scenarios for comparing RL performance."""
        scenarios = []
        
        # Standard care baseline
        scenarios.append({
            'name': 'standard_care_baseline',
            'description': 'Average outcomes from standard diabetes care',
            'target_metrics': {
                'time_in_range': 0.58,
                'severe_hypoglycemia_events': 2,
                'mean_glucose': 165,
                'glucose_cv': 28
            }
        })
        
        # Expert endocrinologist target
        scenarios.append({
            'name': 'expert_care_target',
            'description': 'Target outcomes matching expert endocrinologist',
            'target_metrics': {
                'time_in_range': 0.85,
                'severe_hypoglycemia_events': 0,
                'mean_glucose': 140,
                'glucose_cv': 18
            }
        })
        
        return scenarios
    
    def load_patients(self, dataset: str = "train", use_cache: bool = True) -> List[Dict]:
        """Load patient data for specified dataset."""
        if dataset in self._patients_cache and use_cache:
            return self._patients_cache[dataset]
        
        filepath = os.path.join(self.patient_data_dir, f"{dataset}_patients.json")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Patient dataset '{dataset}' not found. Run setup_datasets() first.")
        
        patients = self.patient_generator.load_patient_data(filepath)
        
        if use_cache:
            self._patients_cache[dataset] = patients
        
        return patients
    
    def load_historical_data(self, patient_id: Optional[str] = None) -> pd.DataFrame:
        """Load historical treatment data."""
        if patient_id:
            # Load specific patient history
            filepath = os.path.join(self.patient_data_dir, f"history_{patient_id}.csv")
            if os.path.exists(filepath):
                return pd.read_csv(filepath)
            else:
                raise FileNotFoundError(f"Historical data for {patient_id} not found")
        else:
            # Load combined historical data
            filepath = os.path.join(self.patient_data_dir, "combined_historical.csv")
            if os.path.exists(filepath):
                return pd.read_csv(filepath)
            else:
                raise FileNotFoundError("Combined historical data not found")
    
    def load_clinical_guidelines(self) -> Dict:
        """Load clinical guidelines."""
        filepath = os.path.join(self.guidelines_dir, "diabetes_guidelines.json")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError("Clinical guidelines not found. Run setup_datasets() first.")
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def get_patient_for_training(self, patient_id: Optional[str] = None, 
                               dataset: str = "train") -> Dict:
        """Get a single patient for training, with proper parameter conversion."""
        patients = self.load_patients(dataset)
        
        if patient_id:
            # Find specific patient
            patient = next((p for p in patients if p['patient_id'] == patient_id), None)
            if not patient:
                raise ValueError(f"Patient {patient_id} not found in {dataset} dataset")
        else:
            # Random patient
            patient = np.random.choice(patients)
        
        # Convert to environment parameters
        env_params = {
            'insulin_sensitivity': patient['diabetes_params']['insulin_sensitivity'],
            'carb_ratio': patient['diabetes_params']['carb_ratio'],
            'basal_glucose': patient['diabetes_params']['basal_glucose'],
            'glucose_variability': patient['diabetes_params']['glucose_variability']
        }
        
        return {
            'patient_info': patient,
            'env_params': env_params
        }
    
    def create_patient_batch(self, batch_size: int, dataset: str = "train") -> List[Dict]:
        """Create a batch of patients for parallel training."""
        patients = self.load_patients(dataset)
        
        if batch_size > len(patients):
            # Sample with replacement if needed
            selected_patients = np.random.choice(patients, batch_size, replace=True)
        else:
            # Sample without replacement
            selected_patients = np.random.choice(patients, batch_size, replace=False)
        
        batch = []
        for patient in selected_patients:
            patient_data = self.get_patient_for_training()
            batch.append(patient_data)
        
        return batch
    
    def evaluate_agent_performance(self, glucose_history: List[float], 
                                 patient_type: str = 'standard_adult') -> Dict:
        """Evaluate agent performance using clinical guidelines."""
        return self.clinical_guidelines.evaluate_glucose_control(
            glucose_history, patient_type
        )
    
    def get_benchmark_targets(self) -> Dict:
        """Get benchmark targets for agent performance."""
        return self.clinical_guidelines.generate_benchmark_data()
    
    def create_dataset_summary(self) -> Dict:
        """Create a comprehensive summary of all datasets."""
        summary = {
            'datasets': {},
            'total_patients': 0,
            'patient_types': {},
            'guidelines': {},
            'creation_date': datetime.now().isoformat()
        }
        
        # Summarize patient datasets
        for dataset in ['train', 'val', 'test']:
            try:
                patients = self.load_patients(dataset)
                
                # Count by patient type
                type_counts = {}
                for patient in patients:
                    ptype = patient['patient_type']
                    type_counts[ptype] = type_counts.get(ptype, 0) + 1
                
                summary['datasets'][dataset] = {
                    'count': len(patients),
                    'patient_types': type_counts
                }
                summary['total_patients'] += len(patients)
                
                # Aggregate patient types
                for ptype, count in type_counts.items():
                    summary['patient_types'][ptype] = summary['patient_types'].get(ptype, 0) + count
                
            except FileNotFoundError:
                summary['datasets'][dataset] = {'count': 0, 'status': 'not_found'}
        
        # Guidelines summary
        try:
            guidelines = self.load_clinical_guidelines()
            summary['guidelines'] = {
                'glucose_targets': len(guidelines.get('glucose_targets', {})),
                'benchmark_types': len(guidelines.get('benchmark_data', {})),
                'available': True
            }
        except FileNotFoundError:
            summary['guidelines'] = {'available': False}
        
        return summary
    
    def visualize_datasets(self):
        """Create visualizations of the datasets."""
        try:
            train_patients = self.load_patients('train')
            
            # Use the existing visualization from patient generator
            self.patient_generator.visualize_patient_population(train_patients)
            
            # Additional visualizations specific to RL training
            self._plot_training_data_distribution(train_patients)
            
        except FileNotFoundError:
            print("❌ No training data found. Run setup_datasets() first.")
    
    def _plot_training_data_distribution(self, patients: List[Dict]):
        """Plot training data distributions relevant to RL."""
        
        # Extract key parameters for RL
        insulin_sens = [p['diabetes_params']['insulin_sensitivity'] for p in patients]
        carb_ratios = [p['diabetes_params']['carb_ratio'] for p in patients]
        glucose_var = [p['diabetes_params']['glucose_variability'] for p in patients]
        hba1c = [p['diabetes_params']['hba1c'] for p in patients]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Insulin sensitivity distribution
        axes[0, 0].hist(insulin_sens, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Insulin Sensitivity Distribution')
        axes[0, 0].set_xlabel('Insulin Sensitivity')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Carb ratio distribution
        axes[0, 1].hist(carb_ratios, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Carb Ratio Distribution')
        axes[0, 1].set_xlabel('Carb Ratio (g/unit)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Glucose variability vs HbA1c
        scatter = axes[1, 0].scatter(glucose_var, hba1c, alpha=0.6, c=insulin_sens, cmap='viridis')
        axes[1, 0].set_xlabel('Glucose Variability')
        axes[1, 0].set_ylabel('HbA1c (%)')
        axes[1, 0].set_title('Glucose Variability vs HbA1c\n(colored by insulin sensitivity)')
        plt.colorbar(scatter, ax=axes[1, 0])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training challenge distribution
        challenge_scores = []
        for patient in patients:
            # Calculate "challenge score" based on multiple factors
            score = 0
            if patient['diabetes_params']['insulin_sensitivity'] < 0.5:
                score += 2  # Insulin resistant
            if patient['diabetes_params']['insulin_sensitivity'] > 1.5:
                score += 2  # Insulin sensitive
            if patient['diabetes_params']['glucose_variability'] > 30:
                score += 3  # High variability
            if patient['diabetes_params']['hba1c'] > 8.0:
                score += 2  # Poor control
            if sum(patient['complications'].values()) > 2:
                score += 1  # Multiple complications
            
            challenge_scores.append(score)
        
        axes[1, 1].hist(challenge_scores, bins=range(11), alpha=0.7, color='coral', edgecolor='black')
        axes[1, 1].set_title('Training Challenge Distribution')
        axes[1, 1].set_xlabel('Challenge Score (0=Easy, 10=Very Hard)')
        axes[1, 1].set_ylabel('Number of Patients')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('RL Training Data Distribution Analysis', y=1.02, fontsize=16)
        plt.show()


def main():
    """Main function to demonstrate data manager usage."""
    print("🏥 RL HEALTHCARE DATA MANAGER DEMO")
    print("=" * 50)
    
    # Initialize data manager
    data_manager = HealthcareDataManager()
    
    # Setup all datasets
    data_manager.setup_datasets(regenerate=False)
    
    # Show dataset summary
    summary = data_manager.create_dataset_summary()
    
    print("\n📊 DATASET SUMMARY")
    print("=" * 30)
    print(f"Total patients: {summary['total_patients']}")
    
    for dataset, info in summary['datasets'].items():
        if 'count' in info:
            print(f"{dataset.title()} set: {info['count']} patients")
    
    print(f"\nPatient type distribution:")
    for ptype, count in summary['patient_types'].items():
        print(f"  {ptype}: {count}")
    
    # Demonstrate data loading
    print("\n🧪 DATA LOADING DEMO")
    print("=" * 30)
    
    # Load a random patient for training
    patient_data = data_manager.get_patient_for_training()
    print(f"Random patient: {patient_data['patient_info']['patient_id']}")
    print(f"Patient type: {patient_data['patient_info']['patient_type']}")
    print(f"Environment parameters: {patient_data['env_params']}")
    
    # Load clinical guidelines
    guidelines = data_manager.load_clinical_guidelines()
    print(f"\nClinical guidelines loaded with {len(guidelines['glucose_targets'])} target groups")
    
    # Demonstrate evaluation
    print("\n📈 EVALUATION DEMO")
    print("=" * 30)
    
    # Simulate some glucose data
    glucose_data = np.random.normal(140, 30, 100).tolist()
    evaluation = data_manager.evaluate_agent_performance(glucose_data)
    
    print(f"Time in Range: {evaluation['time_in_range_percent']:.1f}%")
    print(f"Clinical Assessment: {evaluation['clinical_assessment']}")
    
    # Show benchmarks
    benchmarks = data_manager.get_benchmark_targets()
    print(f"\nBenchmark targets:")
    for care_type, metrics in benchmarks.items():
        if 'time_in_range' in metrics:
            print(f"  {care_type}: {metrics['time_in_range']:.1%} TIR")
    
    # Visualize datasets
    print("\n📊 Creating visualizations...")
    data_manager.visualize_datasets()
    
    print("\n✅ Data manager demo completed!")


if __name__ == "__main__":
    main()