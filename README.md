# fraud-detection-model
Fraud Detection System for Nigerian Digital Payments
🚀 Overview

This project is a machine learning–powered fraud detection system designed to identify suspicious transactions in Nigerian digital payment environments. It uses behavioral and financial patterns to classify transactions as either legitimate or potentially fraudulent in real time through an interactive web interface.

❗ Problem Statement

Nigeria’s digital payment ecosystem is growing rapidly, but fraud is evolving just as fast. Traditional rule-based systems (e.g., fixed transaction limits or blacklists) are no longer sufficient because:

Fraud patterns constantly change
Attackers exploit predictable systems
High transaction volumes make manual monitoring impossible

This creates a need for an adaptive, intelligent system that can detect fraud dynamically without relying solely on static rules.

💡 Solution

This project implements a machine learning model that analyzes transaction-level data such as:

Transaction amount
User income estimate
Average transaction behavior
Device and usage patterns

The system assigns a fraud probability score and flags suspicious transactions for further review.

A Streamlit-based interface allows users to input transaction data and receive instant predictions, making the system both practical and demonstrable.

🧠 Features
Real-time fraud prediction
Machine learning model (Random Forest) for classification
Clean and interactive Streamlit UI
User input validation and formatting
Export of prediction results to Excel
Reset functionality for new transaction entries
Background UI customization for better user experience
🏗️ System Architecture

The system is structured into three main components:

1. Frontend (Streamlit App)
Handles user interaction, input collection, and display of predictions.

2. Model Layer
A trained Random Forest model processes transaction data and outputs predictions.

3. Data Processing Layer
Handles feature preparation, encoding (e.g., device type), and formatting before feeding into the model.

📊 Model Details
Algorithm Used: Random Forest Classifier
Reason for Selection:
Random Forest was chosen for its ability to:
Handle non-linear relationships
Reduce overfitting through ensemble learning
Perform well on structured financial data
Challenges Addressed:
Class imbalance (fraud vs non-fraud)
Feature relevance and scaling
Risk of overfitting on synthetic or limited datasets
Performance:
Model achieved approximately 78% accuracy, indicating reasonable predictive capability but still leaving room for improvement in real-world deployment
