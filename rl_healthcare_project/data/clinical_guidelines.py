#!/usr/bin/env python3
"""
Clinical Guidelines and Reference Data for RL Healthcare Project

This module contains clinical guidelines, treatment protocols, and reference data
for validating and benchmarking RL agent performance in healthcare applications.
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GlucoseTarget:
    """Glucose target ranges for different patient populations."""
    target_min: float
    target_max: float
    severe_low: float
    severe_high: float
    description: str


@dataclass
class InsulinProtocol:
    """Standard insulin dosing protocol."""
    carb_ratio_range: Tuple[float, float]
    correction_factor_range: Tuple[float, float]
    target_glucose: float
    max_dose_per_meal: float


class ClinicalGuidelines:
    """Clinical guidelines and protocols for diabetes management."""
    
    def __init__(self):
        self.glucose_targets = self._define_glucose_targets()
        self.insulin_protocols = self._define_insulin_protocols()
        self.hba1c_targets = self._define_hba1c_targets()
        self.safety_thresholds = self._define_safety_thresholds()
        self.treatment_goals = self._define_treatment_goals()
    
    def _define_glucose_targets(self) -> Dict[str, GlucoseTarget]:
        """Define glucose target ranges by patient type."""
        return {
            'standard_adult': GlucoseTarget(
                target_min=80, target_max=130, 
                severe_low=54, severe_high=250,
                description="Standard adult diabetes management"
            ),
            'elderly': GlucoseTarget(
                target_min=90, target_max=150,
                severe_low=70, severe_high=300,
                description="Elderly patients (>65 years) with relaxed targets"
            ),
            'pediatric': GlucoseTarget(
                target_min=90, target_max=130,
                severe_low=70, severe_high=200,
                description="Pediatric patients (<18 years)"
            ),
            'pregnancy': GlucoseTarget(
                target_min=70, target_max=95,
                severe_low=60, severe_high=140,
                description="Pregnancy (gestational diabetes)"
            ),
            'complications': GlucoseTarget(
                target_min=90, target_max=160,
                severe_low=70, severe_high=300,
                description="Patients with severe complications"
            )
        }
    
    def _define_insulin_protocols(self) -> Dict[str, InsulinProtocol]:
        """Define standard insulin dosing protocols."""
        return {
            'adult_type1': InsulinProtocol(
                carb_ratio_range=(8, 20),
                correction_factor_range=(30, 80),
                target_glucose=120,
                max_dose_per_meal=15
            ),
            'adult_type2': InsulinProtocol(
                carb_ratio_range=(15, 40),
                correction_factor_range=(50, 120),
                target_glucose=130,
                max_dose_per_meal=20
            ),
            'pediatric': InsulinProtocol(
                carb_ratio_range=(15, 30),
                correction_factor_range=(100, 200),
                target_glucose=120,
                max_dose_per_meal=8
            ),
            'elderly': InsulinProtocol(
                carb_ratio_range=(12, 25),
                correction_factor_range=(40, 100),
                target_glucose=140,
                max_dose_per_meal=12
            )
        }
    
    def _define_hba1c_targets(self) -> Dict[str, Dict]:
        """Define HbA1c targets by patient population."""
        return {
            'standard_adult': {'target': 7.0, 'acceptable': 7.5, 'poor_control': 9.0},
            'elderly': {'target': 7.5, 'acceptable': 8.0, 'poor_control': 9.5},
            'pediatric': {'target': 7.0, 'acceptable': 7.5, 'poor_control': 8.5},
            'complications': {'target': 7.5, 'acceptable': 8.0, 'poor_control': 9.0},
            'pregnancy': {'target': 6.0, 'acceptable': 6.5, 'poor_control': 7.0}
        }
    
    def _define_safety_thresholds(self) -> Dict[str, float]:
        """Define safety thresholds for glucose management."""
        return {
            'severe_hypoglycemia': 54,
            'moderate_hypoglycemia': 70,
            'mild_hyperglycemia': 180,
            'severe_hyperglycemia': 250,
            'dka_risk': 300,
            'max_glucose_rise_per_hour': 50,
            'max_glucose_drop_per_hour': 100
        }
    
    def _define_treatment_goals(self) -> Dict[str, Dict]:
        """Define treatment goals and success metrics."""
        return {
            'time_in_range': {
                'excellent': 0.85,  # >85% time in range
                'good': 0.70,       # 70-85% time in range
                'acceptable': 0.50,  # 50-70% time in range
                'poor': 0.50        # <50% time in range
            },
            'hypoglycemia_events': {
                'excellent': 0,     # No severe hypoglycemia
                'good': 1,          # <1 event per month
                'acceptable': 4,     # <4 events per month
                'poor': 4           # >4 events per month
            },
            'glucose_variability': {
                'excellent': 15,    # CV <15%
                'good': 25,         # CV 15-25%
                'acceptable': 35,    # CV 25-35%
                'poor': 35          # CV >35%
            }
        }
    
    def evaluate_glucose_control(self, glucose_history: List[float], 
                               patient_type: str = 'standard_adult') -> Dict:
        """Evaluate glucose control quality based on clinical guidelines."""
        glucose_array = np.array(glucose_history)
        target = self.glucose_targets[patient_type]
        
        # Time in range
        in_range = np.logical_and(glucose_array >= target.target_min, 
                                 glucose_array <= target.target_max)
        time_in_range = np.mean(in_range)
        
        # Hypoglycemic events
        severe_hypo = np.sum(glucose_array < target.severe_low)
        moderate_hypo = np.sum(glucose_array < 70)
        
        # Hyperglycemic events
        severe_hyper = np.sum(glucose_array > target.severe_high)
        moderate_hyper = np.sum(glucose_array > 180)
        
        # Glucose variability
        cv = np.std(glucose_array) / np.mean(glucose_array) * 100
        
        # Mean glucose and estimated HbA1c
        mean_glucose = np.mean(glucose_array)
        estimated_hba1c = (mean_glucose + 46.7) / 28.7
        
        # Clinical assessment
        assessment = self._assess_clinical_quality(
            time_in_range, severe_hypo, cv, estimated_hba1c, patient_type
        )
        
        return {
            'time_in_range': time_in_range,
            'time_in_range_percent': time_in_range * 100,
            'mean_glucose': mean_glucose,
            'estimated_hba1c': estimated_hba1c,
            'glucose_cv': cv,
            'severe_hypoglycemia_events': severe_hypo,
            'moderate_hypoglycemia_events': moderate_hypo,
            'severe_hyperglycemia_events': severe_hyper,
            'moderate_hyperglycemia_events': moderate_hyper,
            'clinical_assessment': assessment,
            'target_range': (target.target_min, target.target_max)
        }
    
    def _assess_clinical_quality(self, tir: float, severe_hypo: int, 
                               cv: float, hba1c: float, patient_type: str) -> str:
        """Assess overall clinical quality of glucose control."""
        hba1c_targets = self.hba1c_targets[patient_type]
        
        # Severe hypoglycemia is always concerning
        if severe_hypo > 0:
            return 'Poor - Severe hypoglycemia detected'
        
        # Assess based on multiple criteria
        if (tir >= 0.85 and hba1c <= hba1c_targets['target'] and cv <= 15):
            return 'Excellent'
        elif (tir >= 0.70 and hba1c <= hba1c_targets['acceptable'] and cv <= 25):
            return 'Good'
        elif (tir >= 0.50 and hba1c <= hba1c_targets['poor_control'] and cv <= 35):
            return 'Acceptable'
        else:
            return 'Poor'
    
    def get_treatment_recommendations(self, glucose_history: List[float],
                                   current_protocol: Dict) -> Dict:
        """Generate treatment recommendations based on current performance."""
        evaluation = self.evaluate_glucose_control(glucose_history)
        recommendations = []
        
        # Time in range recommendations
        if evaluation['time_in_range'] < 0.50:
            recommendations.append({
                'category': 'Time in Range',
                'issue': f"TIR only {evaluation['time_in_range_percent']:.1f}%",
                'recommendation': 'Consider adjusting carb ratios and correction factors',
                'priority': 'High'
            })
        
        # Hypoglycemia recommendations
        if evaluation['severe_hypoglycemia_events'] > 0:
            recommendations.append({
                'category': 'Safety',
                'issue': f"{evaluation['severe_hypoglycemia_events']} severe hypoglycemic events",
                'recommendation': 'Reduce insulin doses, review carb counting accuracy',
                'priority': 'Critical'
            })
        
        # Hyperglycemia recommendations
        if evaluation['severe_hyperglycemia_events'] > 5:
            recommendations.append({
                'category': 'Hyperglycemia',
                'issue': f"{evaluation['severe_hyperglycemia_events']} severe hyperglycemic events",
                'recommendation': 'Increase correction factor, review meal bolus timing',
                'priority': 'High'
            })
        
        # Variability recommendations
        if evaluation['glucose_cv'] > 35:
            recommendations.append({
                'category': 'Variability',
                'issue': f"High glucose variability (CV: {evaluation['glucose_cv']:.1f}%)",
                'recommendation': 'Improve meal timing consistency, review stress factors',
                'priority': 'Medium'
            })
        
        return {
            'evaluation': evaluation,
            'recommendations': recommendations,
            'overall_score': self._calculate_overall_score(evaluation)
        }
    
    def _calculate_overall_score(self, evaluation: Dict) -> float:
        """Calculate overall treatment quality score (0-100)."""
        # Weighted scoring system
        tir_score = min(100, evaluation['time_in_range'] * 100)
        
        # Penalty for hypoglycemia (very important for safety)
        hypo_penalty = evaluation['severe_hypoglycemia_events'] * 20
        hypo_penalty += evaluation['moderate_hypoglycemia_events'] * 5
        
        # Penalty for severe hyperglycemia
        hyper_penalty = evaluation['severe_hyperglycemia_events'] * 2
        
        # Penalty for high variability
        cv_penalty = max(0, evaluation['glucose_cv'] - 15) * 0.5
        
        overall_score = tir_score - hypo_penalty - hyper_penalty - cv_penalty
        return max(0, min(100, overall_score))
    
    def generate_benchmark_data(self) -> Dict:
        """Generate benchmark data for comparing RL agent performance."""
        
        # Standard care outcomes (literature-based)
        standard_care = {
            'time_in_range': 0.58,  # Average TIR for standard care
            'severe_hypoglycemia_rate': 0.15,  # Events per patient-month
            'mean_hba1c': 7.8,
            'glucose_cv': 28
        }
        
        # Intensive management outcomes
        intensive_care = {
            'time_in_range': 0.75,
            'severe_hypoglycemia_rate': 0.08,
            'mean_hba1c': 7.2,
            'glucose_cv': 22
        }
        
        # Expert endocrinologist outcomes
        expert_care = {
            'time_in_range': 0.85,
            'severe_hypoglycemia_rate': 0.03,
            'mean_hba1c': 6.9,
            'glucose_cv': 18
        }
        
        return {
            'standard_care': standard_care,
            'intensive_care': intensive_care,
            'expert_care': expert_care,
            'rl_agent_target': expert_care  # RL agent should aim for expert-level care
        }
    
    def save_guidelines(self, filepath: str):
        """Save clinical guidelines to JSON file."""
        guidelines_data = {
            'glucose_targets': {k: {
                'target_min': v.target_min,
                'target_max': v.target_max,
                'severe_low': v.severe_low,
                'severe_high': v.severe_high,
                'description': v.description
            } for k, v in self.glucose_targets.items()},
            'hba1c_targets': self.hba1c_targets,
            'safety_thresholds': self.safety_thresholds,
            'treatment_goals': self.treatment_goals,
            'benchmark_data': self.generate_benchmark_data()
        }
        
        with open(filepath, 'w') as f:
            json.dump(guidelines_data, f, indent=2)
        
        print(f"Clinical guidelines saved to {filepath}")


def create_validation_scenarios() -> List[Dict]:
    """Create specific validation scenarios for testing RL agents."""
    
    scenarios = [
        {
            'name': 'Dawn Phenomenon',
            'description': 'Early morning glucose rise due to hormonal changes',
            'glucose_pattern': [120, 130, 150, 180, 160, 140],
            'challenge': 'Distinguish from meal effect',
            'expected_action': 'Increase basal insulin or morning correction'
        },
        {
            'name': 'Post-Exercise Hypoglycemia',
            'description': 'Delayed hypoglycemia 4-8 hours after exercise',
            'glucose_pattern': [140, 120, 95, 75, 65, 85],
            'challenge': 'Prevent delayed hypoglycemia',
            'expected_action': 'Reduce insulin or increase carbs preventively'
        },
        {
            'name': 'Sick Day Management',
            'description': 'Increased insulin resistance during illness',
            'glucose_pattern': [180, 220, 250, 280, 260, 240],
            'challenge': 'Manage hyperglycemia without causing DKA',
            'expected_action': 'Increase correction insulin, monitor ketones'
        },
        {
            'name': 'Restaurant Meal',
            'description': 'High-fat, high-carb meal with delayed glucose rise',
            'glucose_pattern': [110, 130, 160, 200, 220, 190],
            'challenge': 'Match insulin timing to delayed absorption',
            'expected_action': 'Extended or dual-wave bolus'
        },
        {
            'name': 'Stress Response',
            'description': 'Glucose elevation due to stress hormones',
            'glucose_pattern': [100, 140, 170, 160, 150, 130],
            'challenge': 'Distinguish from dietary causes',
            'expected_action': 'Temporary correction, address stress'
        }
    ]
    
    return scenarios


def main():
    """Generate and save clinical guidelines and validation data."""
    print("🏥 GENERATING CLINICAL GUIDELINES AND VALIDATION DATA")
    print("=" * 60)
    
    # Create clinical guidelines
    guidelines = ClinicalGuidelines()
    
    # Save guidelines
    import os
    os.makedirs('clinical_guidelines', exist_ok=True)
    guidelines.save_guidelines('clinical_guidelines/diabetes_guidelines.json')
    
    # Create validation scenarios
    scenarios = create_validation_scenarios()
    with open('clinical_guidelines/validation_scenarios.json', 'w') as f:
        json.dump(scenarios, f, indent=2)
    
    print(f"Saved {len(scenarios)} validation scenarios")
    
    # Generate benchmark data
    benchmark_data = guidelines.generate_benchmark_data()
    
    print("\n📊 CLINICAL BENCHMARKS")
    print("=" * 30)
    for care_type, metrics in benchmark_data.items():
        print(f"\n{care_type.replace('_', ' ').title()}:")
        for metric, value in metrics.items():
            if 'rate' in metric:
                print(f"  {metric}: {value:.3f}")
            elif 'time_in_range' in metric:
                print(f"  {metric}: {value:.1%}")
            else:
                print(f"  {metric}: {value:.1f}")
    
    # Example evaluation
    print("\n🧪 EXAMPLE EVALUATION")
    print("=" * 30)
    
    # Simulate glucose data
    np.random.seed(42)
    glucose_data = np.random.normal(150, 40, 288)  # 24 hours of 5-min readings
    glucose_data = np.clip(glucose_data, 50, 400)
    
    evaluation = guidelines.evaluate_glucose_control(glucose_data.tolist())
    recommendations = guidelines.get_treatment_recommendations(glucose_data.tolist(), {})
    
    print(f"Time in Range: {evaluation['time_in_range_percent']:.1f}%")
    print(f"Mean Glucose: {evaluation['mean_glucose']:.1f} mg/dL")
    print(f"Estimated HbA1c: {evaluation['estimated_hba1c']:.1f}%")
    print(f"Clinical Assessment: {evaluation['clinical_assessment']}")
    print(f"Overall Score: {recommendations['overall_score']:.1f}/100")
    
    print("\n✅ Clinical guidelines and validation data created successfully!")


if __name__ == "__main__":
    main()