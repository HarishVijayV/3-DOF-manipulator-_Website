#!/usr/bin/env python3
"""
Patient Data Generator for RL Healthcare Project

This module generates realistic synthetic patient data for training RL agents.
It creates diverse patient populations with clinically realistic parameters.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns


class PatientDataGenerator:
    """Generate realistic synthetic patient data for healthcare RL training."""
    
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        self.patient_types = {
            'type1_child': {
                'age_range': (8, 17),
                'weight_range': (25, 70),
                'insulin_sensitivity_range': (0.8, 1.5),
                'carb_ratio_range': (8, 15),
                'basal_glucose_range': (90, 120),
                'glucose_variability_range': (15, 25),
                'activity_level': (0.6, 1.0)
            },
            'type1_adult': {
                'age_range': (18, 65),
                'weight_range': (50, 120),
                'insulin_sensitivity_range': (0.5, 1.2),
                'carb_ratio_range': (10, 20),
                'basal_glucose_range': (85, 115),
                'glucose_variability_range': (10, 20),
                'activity_level': (0.3, 0.8)
            },
            'type1_elderly': {
                'age_range': (65, 85),
                'weight_range': (45, 100),
                'insulin_sensitivity_range': (0.3, 0.8),
                'carb_ratio_range': (12, 25),
                'basal_glucose_range': (90, 130),
                'glucose_variability_range': (8, 15),
                'activity_level': (0.1, 0.5)
            },
            'type2_controlled': {
                'age_range': (30, 70),
                'weight_range': (60, 150),
                'insulin_sensitivity_range': (0.3, 0.7),
                'carb_ratio_range': (15, 30),
                'basal_glucose_range': (100, 140),
                'glucose_variability_range': (12, 18),
                'activity_level': (0.2, 0.6)
            },
            'type2_uncontrolled': {
                'age_range': (35, 75),
                'weight_range': (70, 180),
                'insulin_sensitivity_range': (0.2, 0.5),
                'carb_ratio_range': (20, 40),
                'basal_glucose_range': (120, 180),
                'glucose_variability_range': (20, 35),
                'activity_level': (0.1, 0.4)
            }
        }
    
    def generate_patient_population(self, n_patients: int = 100) -> List[Dict]:
        """Generate a diverse population of synthetic patients."""
        patients = []
        
        # Distribution of patient types
        type_distribution = {
            'type1_child': 0.15,
            'type1_adult': 0.25,
            'type1_elderly': 0.10,
            'type2_controlled': 0.30,
            'type2_uncontrolled': 0.20
        }
        
        for i in range(n_patients):
            # Select patient type based on distribution
            patient_type = np.random.choice(
                list(type_distribution.keys()),
                p=list(type_distribution.values())
            )
            
            # Generate patient parameters
            patient = self._generate_single_patient(patient_type, i)
            patients.append(patient)
        
        return patients
    
    def _generate_single_patient(self, patient_type: str, patient_id: int) -> Dict:
        """Generate a single patient with realistic parameters."""
        params = self.patient_types[patient_type]
        
        # Basic demographics
        age = np.random.uniform(*params['age_range'])
        weight = np.random.uniform(*params['weight_range'])
        
        # Diabetes-specific parameters
        insulin_sensitivity = np.random.uniform(*params['insulin_sensitivity_range'])
        carb_ratio = np.random.uniform(*params['carb_ratio_range'])
        basal_glucose = np.random.uniform(*params['basal_glucose_range'])
        glucose_variability = np.random.uniform(*params['glucose_variability_range'])
        activity_level = np.random.uniform(*params['activity_level'])
        
        # Additional realistic factors
        bmi = weight / ((170 + np.random.normal(0, 15)) / 100) ** 2  # Rough BMI calculation
        hba1c = self._calculate_hba1c(basal_glucose, glucose_variability)
        
        # Meal patterns (affects training)
        meal_schedule = self._generate_meal_schedule()
        
        # Complications and comorbidities
        complications = self._generate_complications(patient_type, age)
        
        patient = {
            'patient_id': f'patient_{patient_id:04d}',
            'patient_type': patient_type,
            'demographics': {
                'age': round(age, 1),
                'weight': round(weight, 1),
                'bmi': round(bmi, 1),
                'gender': np.random.choice(['M', 'F'])
            },
            'diabetes_params': {
                'insulin_sensitivity': round(insulin_sensitivity, 3),
                'carb_ratio': round(carb_ratio, 1),
                'basal_glucose': round(basal_glucose, 1),
                'glucose_variability': round(glucose_variability, 1),
                'hba1c': round(hba1c, 1),
                'diabetes_duration': round(np.random.exponential(8), 1)
            },
            'lifestyle': {
                'activity_level': round(activity_level, 2),
                'meal_schedule': meal_schedule,
                'stress_level': round(np.random.uniform(0.1, 0.8), 2),
                'sleep_quality': round(np.random.uniform(0.3, 1.0), 2)
            },
            'complications': complications,
            'created_at': datetime.now().isoformat()
        }
        
        return patient
    
    def _calculate_hba1c(self, basal_glucose: float, variability: float) -> float:
        """Calculate estimated HbA1c based on glucose parameters."""
        # Simplified relationship: higher glucose and variability -> higher HbA1c
        avg_glucose = basal_glucose + np.random.normal(0, variability/3)
        hba1c = (avg_glucose + 46.7) / 28.7  # ADAG formula approximation
        return np.clip(hba1c, 5.0, 15.0)
    
    def _generate_meal_schedule(self) -> Dict:
        """Generate realistic meal timing and carb patterns."""
        return {
            'breakfast_time': round(np.random.normal(7.5, 1.0), 1),  # 7:30 AM ± 1 hour
            'lunch_time': round(np.random.normal(12.5, 1.0), 1),    # 12:30 PM ± 1 hour
            'dinner_time': round(np.random.normal(18.5, 1.5), 1),   # 6:30 PM ± 1.5 hours
            'snack_frequency': round(np.random.uniform(0, 3), 1),   # 0-3 snacks per day
            'carb_consistency': round(np.random.uniform(0.3, 1.0), 2)  # Meal consistency
        }
    
    def _generate_complications(self, patient_type: str, age: float) -> Dict:
        """Generate diabetes complications based on patient type and age."""
        base_risk = 0.1 if 'type1' in patient_type else 0.2
        age_factor = max(0, (age - 40) / 40)  # Increased risk with age
        
        complications = {
            'retinopathy': np.random.random() < (base_risk + age_factor * 0.3),
            'nephropathy': np.random.random() < (base_risk + age_factor * 0.2),
            'neuropathy': np.random.random() < (base_risk + age_factor * 0.25),
            'cardiovascular': np.random.random() < (base_risk * 0.5 + age_factor * 0.4)
        }
        
        return complications
    
    def generate_historical_data(self, patient: Dict, days: int = 90) -> pd.DataFrame:
        """Generate historical glucose and treatment data for a patient."""
        records = []
        current_date = datetime.now() - timedelta(days=days)
        
        # Patient parameters
        params = patient['diabetes_params']
        lifestyle = patient['lifestyle']
        
        for day in range(days):
            # Generate daily patterns (breakfast, lunch, dinner)
            for meal_time, meal_name in [(7.5, 'breakfast'), (12.5, 'lunch'), (18.5, 'dinner')]:
                
                # Add some variation to meal times
                actual_time = meal_time + np.random.normal(0, 0.5)
                
                # Pre-meal glucose
                pre_meal_glucose = np.random.normal(
                    params['basal_glucose'], 
                    params['glucose_variability']
                )
                
                # Carb intake (varies by meal and patient)
                carb_multipliers = {'breakfast': 0.8, 'lunch': 1.2, 'dinner': 1.0}
                base_carbs = np.random.normal(45, 15) * carb_multipliers[meal_name]
                carb_intake = max(10, base_carbs * lifestyle['carb_consistency'])
                
                # Insulin dose (based on carb ratio and correction)
                insulin_for_carbs = carb_intake / params['carb_ratio']
                correction_insulin = max(0, (pre_meal_glucose - 120) / 50)
                total_insulin = insulin_for_carbs + correction_insulin
                
                # Post-meal glucose simulation
                glucose_rise = carb_intake * 3
                glucose_drop = total_insulin * params['insulin_sensitivity'] * 15
                post_meal_glucose = pre_meal_glucose + glucose_rise - glucose_drop
                post_meal_glucose += np.random.normal(0, params['glucose_variability'] / 2)
                
                # Activity effect
                if np.random.random() < lifestyle['activity_level']:
                    post_meal_glucose -= np.random.uniform(10, 30)
                
                # Ensure realistic bounds
                pre_meal_glucose = np.clip(pre_meal_glucose, 50, 400)
                post_meal_glucose = np.clip(post_meal_glucose, 50, 400)
                
                record = {
                    'patient_id': patient['patient_id'],
                    'timestamp': current_date + timedelta(hours=actual_time),
                    'meal_type': meal_name,
                    'pre_meal_glucose': round(pre_meal_glucose, 1),
                    'post_meal_glucose': round(post_meal_glucose, 1),
                    'carb_intake': round(carb_intake, 1),
                    'insulin_dose': round(total_insulin, 1),
                    'activity_level': lifestyle['activity_level'],
                    'stress_level': lifestyle['stress_level']
                }
                
                records.append(record)
            
            current_date += timedelta(days=1)
        
        return pd.DataFrame(records)
    
    def save_patient_data(self, patients: List[Dict], filepath: str):
        """Save patient data to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(patients, f, indent=2)
        
        print(f"Saved {len(patients)} patients to {filepath}")
    
    def load_patient_data(self, filepath: str) -> List[Dict]:
        """Load patient data from JSON file."""
        with open(filepath, 'r') as f:
            patients = json.load(f)
        
        print(f"Loaded {len(patients)} patients from {filepath}")
        return patients
    
    def create_patient_summary(self, patients: List[Dict]) -> pd.DataFrame:
        """Create a summary DataFrame of patient characteristics."""
        summary_data = []
        
        for patient in patients:
            summary = {
                'patient_id': patient['patient_id'],
                'patient_type': patient['patient_type'],
                'age': patient['demographics']['age'],
                'weight': patient['demographics']['weight'],
                'bmi': patient['demographics']['bmi'],
                'insulin_sensitivity': patient['diabetes_params']['insulin_sensitivity'],
                'carb_ratio': patient['diabetes_params']['carb_ratio'],
                'basal_glucose': patient['diabetes_params']['basal_glucose'],
                'hba1c': patient['diabetes_params']['hba1c'],
                'activity_level': patient['lifestyle']['activity_level'],
                'complications_count': sum(patient['complications'].values())
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def visualize_patient_population(self, patients: List[Dict]):
        """Create visualizations of the patient population characteristics."""
        df = self.create_patient_summary(patients)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Patient type distribution
        axes[0, 0].pie(df['patient_type'].value_counts().values, 
                       labels=df['patient_type'].value_counts().index,
                       autopct='%1.1f%%')
        axes[0, 0].set_title('Patient Type Distribution')
        
        # Age distribution by type
        sns.boxplot(data=df, x='patient_type', y='age', ax=axes[0, 1])
        axes[0, 1].set_title('Age Distribution by Patient Type')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Insulin sensitivity vs Carb ratio
        scatter = axes[0, 2].scatter(df['insulin_sensitivity'], df['carb_ratio'], 
                                   c=df['age'], cmap='viridis', alpha=0.6)
        axes[0, 2].set_xlabel('Insulin Sensitivity')
        axes[0, 2].set_ylabel('Carb Ratio')
        axes[0, 2].set_title('Insulin Sensitivity vs Carb Ratio (colored by age)')
        plt.colorbar(scatter, ax=axes[0, 2])
        
        # HbA1c distribution
        axes[1, 0].hist(df['hba1c'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=7.0, color='red', linestyle='--', label='Target <7%')
        axes[1, 0].set_xlabel('HbA1c (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('HbA1c Distribution')
        axes[1, 0].legend()
        
        # BMI vs Activity Level
        scatter2 = axes[1, 1].scatter(df['bmi'], df['activity_level'], 
                                     c=df['complications_count'], cmap='Reds', alpha=0.6)
        axes[1, 1].set_xlabel('BMI')
        axes[1, 1].set_ylabel('Activity Level')
        axes[1, 1].set_title('BMI vs Activity Level (colored by complications)')
        plt.colorbar(scatter2, ax=axes[1, 1])
        
        # Glucose variability by patient type
        glucose_var_data = []
        for patient in patients:
            glucose_var_data.append({
                'patient_type': patient['patient_type'],
                'glucose_variability': patient['diabetes_params']['glucose_variability']
            })
        glucose_df = pd.DataFrame(glucose_var_data)
        
        sns.violinplot(data=glucose_df, x='patient_type', y='glucose_variability', ax=axes[1, 2])
        axes[1, 2].set_title('Glucose Variability by Patient Type')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()


def main():
    """Generate and save patient data for the RL healthcare project."""
    print("🏥 GENERATING PATIENT DATA FOR RL HEALTHCARE PROJECT")
    print("=" * 60)
    
    # Initialize generator
    generator = PatientDataGenerator(random_seed=42)
    
    # Generate patient populations
    print("Generating patient populations...")
    
    # Training population (large, diverse)
    train_patients = generator.generate_patient_population(n_patients=200)
    
    # Validation population (smaller, for testing)
    val_patients = generator.generate_patient_population(n_patients=50)
    
    # Test population (specific challenging cases)
    test_patients = generator.generate_patient_population(n_patients=30)
    
    # Save patient data
    os.makedirs('synthetic_patients', exist_ok=True)
    
    generator.save_patient_data(train_patients, 'synthetic_patients/train_patients.json')
    generator.save_patient_data(val_patients, 'synthetic_patients/val_patients.json')
    generator.save_patient_data(test_patients, 'synthetic_patients/test_patients.json')
    
    # Generate historical data for a subset
    print("Generating historical treatment data...")
    
    historical_data_all = []
    for i, patient in enumerate(train_patients[:20]):  # First 20 patients
        historical_df = generator.generate_historical_data(patient, days=90)
        historical_data_all.append(historical_df)
        
        if i < 3:  # Save individual files for first 3 patients
            historical_df.to_csv(
                f'synthetic_patients/historical_data_{patient["patient_id"]}.csv', 
                index=False
            )
    
    # Combine all historical data
    combined_historical = pd.concat(historical_data_all, ignore_index=True)
    combined_historical.to_csv('synthetic_patients/combined_historical_data.csv', index=False)
    
    # Create and save patient summaries
    train_summary = generator.create_patient_summary(train_patients)
    train_summary.to_csv('synthetic_patients/train_patient_summary.csv', index=False)
    
    # Visualize patient population
    print("Creating patient population visualizations...")
    generator.visualize_patient_population(train_patients)
    
    # Print statistics
    print("\n📊 PATIENT DATA STATISTICS")
    print("=" * 40)
    print(f"Training patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"Test patients: {len(test_patients)}")
    print(f"Historical records: {len(combined_historical)}")
    
    print("\nPatient type distribution (training set):")
    type_counts = train_summary['patient_type'].value_counts()
    for ptype, count in type_counts.items():
        print(f"  {ptype}: {count} ({count/len(train_patients)*100:.1f}%)")
    
    print(f"\nHbA1c statistics (training set):")
    print(f"  Mean: {train_summary['hba1c'].mean():.1f}%")
    print(f"  Std: {train_summary['hba1c'].std():.1f}%")
    print(f"  Patients with HbA1c >7%: {(train_summary['hba1c'] > 7).sum()}")
    
    print("\n✅ Patient data generation completed successfully!")
    print("Files saved in: synthetic_patients/")


if __name__ == "__main__":
    main()