# Project# Automated SOC Alert Triage Using Machine Learning

## Overview
This project presents a machine learning-based framework for automated alert triage in Security Operations Centres (SOC). The system is designed to help SOC analysts manage large volumes of alerts, reduce false positives, and prioritise security threats efficiently.

Machine learning models including Random Forest, XGBoost, and Long Short-Term Memory (LSTM) networks are implemented and evaluated on public cybersecurity datasets such as UNSW NB15, CICIDS 2017, CICIDS 2018, and NSL KDD.

## Features
- Automated classification of cybersecurity alerts
- Alert prioritisation to support SOC decision-making
- Reduction of false positives and alert fatigue
- Performance evaluation using accuracy, precision, recall, F1-score, ROC AUC, confusion matrix
- Contextual mapping inspired by MITRE ATT&CK framework

## Technologies Used
- Python
- Scikit-learn
- XGBoost
- TensorFlow / Keras (for LSTM)
- Pandas & NumPy for data processing
- Matplotlib & Seaborn for visualization

## Datasets
- [UNSW NB15](https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/)
- [CICIDS 2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [CICIDS 2018](https://www.unb.ca/cic/datasets/ids-2018.html)
- [NSL KDD](https://www.unb.ca/cic/datasets/nsl.html)

## Getting Started
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/soc-alert-triage.git
