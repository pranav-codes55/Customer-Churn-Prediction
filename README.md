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
├── customer_churn.py                    # Data processing & model training
├── streamlit_app.py                     # Interactive dashboard
├── customer_churn_prediction.csv        # Dataset
├── requirements.txt                     # Project dependencies
├── README.md                            # Main documentation
├── MySQL/                               # Database scripts & schema
│   ├── README.md
│   ├── schema.sql                       # Database structure
│   ├── queries.sql                      # SQL queries
│   └── data_import.sql                  # Data import scripts
├── Tableau/                             # Visualizations & dashboards
│   ├── README.md
│   ├── churn_analysis.twbx
│   ├── customer_insights.twbx
│   └── PDF_exports/
└── .gitignore                           # Git ignore rules
```

## Technologies Used

- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning
- **Streamlit** - Web dashboard framework
- **MySQL** - Database management & queries
- **Tableau** - Data visualization & BI dashboards

## Project Components

### 1. Python Machine Learning (customer_churn.py)
Core ML pipeline that:
- Loads and cleans customer data
- Preprocesses categorical and numerical features
- Trains a Random Forest classifier
- Generates model evaluation metrics

### 2. Interactive Web Dashboard (streamlit_app.py)
User-friendly interface for:
- Uploading custom datasets
- Training models with configurable parameters
- Making predictions on new customers
- Viewing real-time performance metrics

### 3. MySQL Database (MySQL/)
Contains:
- Database schema with customer and churn tables
- Historical queries for data analysis
- Scripts to prepare and import data
- Connection patterns for integration

### 4. Tableau Visualizations (Tableau/)
Business intelligence dashboards featuring:
- Churn analysis by customer segment
- Trend analysis over time
- Customer behavior insights
- Interactive filters and drill-downs

## Workflow

```
Data (CSV) → MySQL Database → Python ML Model → Predictions
                     ↓
            Tableau Dashboards ← SQL Queries
                     ↓
            Streamlit Dashboard ← ML Model
```

## Model Performance

The Random Forest model provides metrics including:
- Accuracy
- Precision & Recall
- Confusion Matrix
- Classification Report

## License

This project is open source and available under the MIT License.
