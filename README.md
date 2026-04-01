# Customer Churn Prediction

A machine learning project to predict customer churn using Random Forest classification and an interactive Streamlit dashboard.

## Project Overview

This project analyzes customer data and builds a predictive model to identify customers likely to churn. It includes:
- **Data Processing & Model Training** (`customer_churn.py`) - Cleans data, trains a Random Forest classifier, and evaluates performance
- **Interactive Dashboard** (`streamlit_app.py`) - Web interface to train models, make predictions, and view analytics

## Dataset

The project uses `customer_churn_prediction.csv` containing customer information with a target variable indicating churn status.

## Features

- Automated data cleaning (handle missing values, duplicates)
- One-hot encoding for categorical variables
- Feature scaling for numerical data
- Random Forest classification with evaluation metrics
- Interactive Streamlit dashboard for model training and predictions
- Real-time performance metrics and visualizations

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Run the Training Script
```bash
python customer_churn.py
```

### Launch the Dashboard
```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Project Structure

```
.
├── customer_churn.py              # Data processing & model training
├── streamlit_app.py               # Interactive dashboard
├── customer_churn_prediction.csv  # Dataset
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

## Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning
- **Streamlit** - Web dashboard framework

## Model Performance

The Random Forest model provides metrics including:
- Accuracy
- Precision & Recall
- Confusion Matrix
- Classification Report

## License

This project is open source and available under the MIT License.
