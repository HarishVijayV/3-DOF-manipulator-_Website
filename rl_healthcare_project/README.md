# AI-Driven Drug Dosage Optimization using Reinforcement Learning

This project implements multiple reinforcement learning algorithms to optimize drug dosage for treating diabetes, hypertension, and depression. It provides an end-to-end AI pipeline that can be extended into a full-stack web application.

## 🎯 Project Overview

Traditional drug dosage determination relies on fixed protocols, but patient responses vary significantly. This project uses RL to learn optimal dosage strategies by:
- Monitoring patient vital signs and biomarkers
- Adjusting dosages dynamically based on patient response  
- Minimizing side effects while maximizing therapeutic benefit

## 🏥 Disease Applications

### 1. Diabetes Management (DQN)
- **State**: Blood glucose, meal timing, carb intake, previous insulin, exercise
- **Action**: Insulin dosage (low/medium/high)
- **Goal**: Maintain glucose in target range (80-120 mg/dL)

### 2. Hypertension Treatment (PPO) 
- **State**: Blood pressure, heart rate, demographics, medication history
- **Action**: Continuous medication dosage adjustment
- **Goal**: Achieve target blood pressure with minimal side effects

### 3. Depression Treatment (A3C)
- **State**: Depression scores, side effects, treatment duration
- **Action**: Dosage adjustments and medication switches
- **Goal**: Improve depression scores while minimizing adverse effects

## 🧠 RL Algorithms

- **DQN (Deep Q-Network)**: Discrete action spaces, experience replay
- **PPO (Proximal Policy Optimization)**: Continuous control, policy gradients
- **A3C (Advantage Actor-Critic)**: Asynchronous learning, multi-patient scenarios

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd rl_healthcare_project
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Test the diabetes environment**
```bash
cd environments
python diabetes_env.py
```

4. **Train the DQN agent**
```bash
python train_diabetes_dqn.py --episodes 1000 --eval_freq 100
```

### Training Parameters

```bash
python train_diabetes_dqn.py \
    --episodes 2000 \
    --lr 0.001 \
    --batch_size 64 \
    --memory_size 20000 \
    --eval_freq 200
```

## 📁 Project Structure

```
rl_healthcare_project/
├── environments/           # Disease simulation environments
│   ├── diabetes_env.py    # Diabetes management environment
│   ├── hypertension_env.py # Blood pressure control (TODO)
│   └── depression_env.py   # Mental health treatment (TODO)
├── algorithms/            # RL algorithm implementations
│   ├── dqn/              # Deep Q-Network
│   ├── ppo/              # Proximal Policy Optimization (TODO)
│   └── a3c/              # Advantage Actor-Critic (TODO)
├── models/               # Trained models and patient simulators
├── evaluation/           # Metrics and visualization tools
├── web_app/             # Full-stack web application (TODO)
├── notebooks/           # Jupyter notebooks for analysis
├── data/                # Synthetic patient data
├── tests/               # Unit tests
├── train_diabetes_dqn.py # Main training script
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## 📊 Results and Evaluation

The training script automatically generates:

- **Training plots**: Rewards, losses, convergence metrics
- **Patient evaluation**: Performance across diverse patient populations
- **Clinical metrics**: Time in range, hypoglycemic/hyperglycemic events
- **Model checkpoints**: Saved trained models for deployment

### Key Metrics

- **Time in Range (TIR)**: Percentage of time glucose stays in 80-120 mg/dL
- **Hypoglycemic Events**: Episodes below 70 mg/dL  
- **Hyperglycemic Events**: Episodes above 180 mg/dL
- **Mean Absolute Error**: Deviation from target glucose levels

## 🔬 Customization

### Creating New Patient Profiles

```python
patient_params = {
    'insulin_sensitivity': 1.2,  # Individual response to insulin
    'carb_ratio': 12.0,         # Carbs per insulin unit
    'basal_glucose': 95.0,      # Baseline glucose level
    'glucose_variability': 8.0   # Natural glucose fluctuation
}

env = DiabetesEnv(patient_params)
```

### Modifying Reward Functions

Edit the `_calculate_reward()` method in `environments/diabetes_env.py`:

```python
def _calculate_reward(self, glucose: float, insulin: float) -> float:
    # Custom reward logic here
    if 80 <= glucose <= 120:
        return 10.0  # In target range
    else:
        # Penalty based on distance from target
        return -0.1 * (distance_from_target ** 2)
```

### Hyperparameter Tuning

Key parameters to experiment with:
- **Learning rate**: 1e-4 to 1e-2
- **Network architecture**: Hidden layer sizes, depth
- **Exploration**: Epsilon decay schedule
- **Experience replay**: Buffer size, batch size

## 🌐 Web Application (Roadmap)

The full-stack component will include:

### Backend (FastAPI)
- REST API for model inference
- Patient data management
- Real-time dosage recommendations
- Treatment history tracking

### Frontend (React)
- Healthcare dashboard
- Real-time glucose monitoring
- Dosage recommendation interface
- Patient management system

### Database (PostgreSQL)
- Patient profiles and history
- Treatment outcomes
- Model performance metrics

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Test specific components:
```bash
pytest tests/test_diabetes_env.py
pytest tests/test_dqn_agent.py
```

## 📈 Performance Optimization

### GPU Acceleration
The models automatically use CUDA if available:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Distributed Training
For larger experiments, use Ray RLlib:
```python
import ray
from ray import tune

# Distributed hyperparameter tuning
ray.init()
tune.run(...)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Academic Usage

This project is designed for educational purposes and research. Key academic contributions:

- **Multi-algorithm comparison** in healthcare applications
- **Patient-centric RL** with diverse population modeling
- **Clinical validation metrics** beyond standard RL rewards
- **End-to-end pipeline** from simulation to deployment

### Citation
If you use this project in your research, please cite:
```
@misc{rl_healthcare_dosage,
    title={AI-Driven Drug Dosage Optimization using Reinforcement Learning},
    author={Your Name},
    year={2024},
    url={https://github.com/your-username/rl_healthcare_project}
}
```

## ⚠️ Disclaimer

This project is for educational and research purposes only. It should not be used for actual medical decision-making without proper clinical validation and regulatory approval.

## 📚 References

1. Sutton & Barto - Reinforcement Learning: An Introduction
2. Mnih et al. - Human-level control through deep reinforcement learning
3. Schulman et al. - Proximal Policy Optimization Algorithms
4. Clinical guidelines for diabetes management (ADA/EASD)

## 📧 Contact

For questions or collaboration opportunities:
- Email: [your.email@example.com]
- GitHub: [@your-username]

---

**Happy Learning! 🎓🏥🤖**