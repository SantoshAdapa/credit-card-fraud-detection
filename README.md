# credit-card-fraud-detection
A deep learning model to detect fraudulent credit card transactions with high precision using real-world data.

# Credit Card Fraud Detection (CCFD) using RTAHC Model
## Project Description:
Credit card fraud detection (CCFD) plays a pivotal role in maintaining financial security. Traditional models often face challenges in detecting fraud due to the following:
1. **Feature Distribution Differences**: Transaction patterns change over time (concept drift), making fraud detection increasingly difficult.
2. **Insufficient Feature Representation**: Fraudulent transactions are often concealed among legitimate ones, requiring advanced feature extraction methods.

This project proposes a novel deep learning-based solution to address these challenges using **Deep Reinforcement Learning (RL)** and **Attention Mechanisms**.

### Proposed Model: RTAHC
The **RTAHC (Reinforcement Learning-based Transaction Fraud Detection with Attention Mechanism)** model consists of two main components:

#### A. Selection Distribution Generator (SDG)
- Uses **Reinforcement Learning** combined with an **Attention Mechanism** to select the most relevant transactions for training.
- The **SDG** rewards the selection of valuable historical transactions.
- Implemented using a **Multilayer Perceptron (MLP)**.

#### B. Transaction Fraud Detector (TFD)
- A **Convolutional Neural Network (CNN)** with an Attention Mechanism.
- **CNN layers** act as a **feature extractor**, while the final **Softmax layer** classifies the transactions.
- **Key Innovation**: Joint training of SDG and TFD, which minimizes feature distribution differences and improves fraud detection.

### Implementation Strategy
1. Data Preparation:
- Use the Kaggle Credit Card Fraud Dataset for training.
- Preprocess and normalize transaction features.

2. RTAHC Model Implementation:
- Train SDG using Reinforcement Learning to choose optimal training data.
- Train TFD using CNN with Attention Mechanism.
- Perform joint training for SDG and TFD using policy gradients from reinforcement learning.

3. Evaluation & Performance Metrics:
- Measure performance using Precision, Recall, F1-score, and AUC.
- Compare results against baseline models such as Random Forest, XGBoost, and LSTM.

## Dataset Overview
The dataset used in this project is the **Credit Card Fraud Detection** dataset from Kaggle. It contains the following:
- **284,807** transaction records
- **30 anonymized features** (V1 to V28) derived from **Principal Component Analysis (PCA)**
- **2 additional features**:
  - **Time**: Time elapsed from the first transaction to the current transaction.
  - **Amount**: The transaction amount.
- **Target Variable (Class)**:
  - **0**: Legitimate transactions.
  - **1**: Fraudulent transactions.

## Technologies Used
- Python
- Pandas: Data manipulation and analysis
- NumPy: Numerical operations
- TensorFlow/PyTorch: Deep learning models
- Matplotlib/Seaborn: Data visualization

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
2. Install the necessary dependencies:
   ```nginx
   pip install -r requirements.txt
3. Run the Jupyter Notebook:
   ```nginx
   jupyter notebook fraud_detection.ipynb

## Results
This model outperforms traditional algorithms like **Random Forest**, **XGBoost**, and **LSTM** in detecting fraudulent transactions, demonstrating the effectiveness of combining **reinforcement learning** with **attention mechanisms**.
