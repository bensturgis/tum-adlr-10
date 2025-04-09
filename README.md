# Active Learning for Dynamical Systems using Bayesian Neural Networks

This project explores data-efficient learning of unknown dynamical systems using active learning with Bayesian Neural Networks (BNNs). When collecting real-world data is costly, actively selecting informative samples can significantly improve sample efficiency.

## Key Features

- **Bayesian Neural Network Model**: Learns the system dynamics while quantifying uncertainty.
- **Active Learning Strategies**:
  - Random Sampling Shooting with Model Predictive Control
  - Soft Actor Critic (SAC) policy optimization
- **Feature Expansion**: Ensures uncertainty estimates remain invariant to input norm.
- **Custom Environments**:
  - Mass-Spring-Damper System
  - 2D Reacher Task

## Evaluation

We evaluate performance based on:
- One-step and multi-step prediction error
- State space exploration efficiency

## Results

Our methods show superior sample efficiency compared to random exploration, thanks to uncertainty-guided data collection.

## Code and Experiments

You can find the full implementation and experiments in this repository.

TUM-ADLR-10/
├── environments/               # Custom environments
├── experiments/                # Scripts, results and saved data for experiments
├── metrics/                    # Evaluation metrics
├── models/                     # Neural network model definitions
├── sampling_methods/           # Active learning exploration algorithms
├── utils/                      # Utility functions
└── weights/                    # Trained model weights

## Poster and Paper
This repository is based on our course project for TUM ADLR course. For presentation poster and report paper, see folder presentation/ .
