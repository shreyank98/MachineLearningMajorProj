# Loan Approval Prediction - Machine Learning Project

## 📋 Project Overview
This project implements a comprehensive machine learning pipeline to predict loan approval status using ensemble methods. The system analyzes applicant data including income, credit history, and demographics to make informed loan approval decisions with a focus on minimizing risk.

## 🎯 Key Features
- **Advanced Data Preprocessing**: Handles missing values, outliers, and class imbalance
- **Feature Engineering**: Creates derived features like "Ability to Pay" ratio
- **Ensemble Learning**: Implements Voting and Stacking classifiers combining Random Forest and XGBoost
- **SMOTE Oversampling**: Balances the dataset to improve minority class prediction
- **Threshold Optimization**: Fine-tunes decision threshold for risk-adjusted predictions
- **Comprehensive Visualization**: Decision boundaries, feature importance, and error analysis

## 📊 Dataset
The project uses a loan application dataset (`loan-train.csv`) containing:
- **Applicant Demographics**: Gender, Marital Status, Dependents, Education
- **Financial Data**: Applicant Income, Coapplicant Income, Loan Amount
- **Loan Details**: Loan Term, Credit History, Property Area
- **Target Variable**: Loan Status (Approved/Rejected)

## 🛠️ Technologies Used
- **Python 3.x**
- **Core Libraries**:
  - `pandas` - Data manipulation
  - `numpy` - Numerical computations
  - `scikit-learn` - Machine learning algorithms
  - `xgboost` - Gradient boosting
  - `imblearn` - SMOTE oversampling
- **Visualization**:
  - `matplotlib`
  - `seaborn`
- **Model Persistence**:
  - `joblib`

## 🔄 Machine Learning Pipeline

### 1. Data Loading & Exploration
- Load dataset and inspect shape, types, and missing values
- Analyze target variable distribution

### 2. Data Cleaning & Preprocessing
- Drop non-predictive columns (Loan_ID)
- Handle "Dependents" field (convert "3+" to 3)
- Fill missing categorical values with mode
- Fill missing numerical values with median
- Verify data integrity

### 3. Feature Engineering
- **Total Income**: Sum of applicant and coapplicant income
- **Log Transformations**: Apply to LoanAmount and Total_Income to handle skewness
- **Ability to Pay**: Golden feature = Total_Income_Log - LoanAmount_Log
- Remove raw features to reduce multicollinearity

### 4. Encoding & Splitting
- Label encode categorical variables (Gender, Married, Education, etc.)
- Split data into 80% training and 20% testing sets
- Preserve random state for reproducibility

### 5. Handling Class Imbalance
- Apply **SMOTE** (Synthetic Minority Over-sampling Technique)
- Balance training set to 50-50 distribution
- Prevents model bias toward majority class

### 6. Model Training & Hyperparameter Tuning
Three models with RandomizedSearchCV:
- **Decision Tree Classifier**
- **Random Forest Classifier** ⭐
- **XGBoost Classifier** ⭐

Optimization metric: `f1_macro` (balanced F1 score)

### 7. Best Model - Random Forest
```python
RandomForestClassifier(
    bootstrap=False,
    max_depth=None,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=158,
    random_state=42
)
```

### 8. Advanced Ensemble Methods

#### Voting Classifier (Soft Voting)
Combines Random Forest and XGBoost by averaging their probability predictions.

#### Stacking Classifier
- **Base Models**: Random Forest + XGBoost
- **Meta-Model**: Logistic Regression
- Learns optimal way to combine base model predictions

### 9. Threshold Optimization
- Default threshold: 0.5
- Adjusted threshold: **0.6** (safer, catches more risky loans)
- Trade-off: Higher precision for rejections at cost of some approvals

### 10. Model Deployment
- Save best model using `joblib`
- Create prediction function with complete preprocessing pipeline
- Test with ideal and risky candidate profiles

## 📈 Model Performance

### Feature Importance
The model identifies key drivers:
1. **Credit_History** - Most critical factor
2. **Total_Income_Log** - Ability to repay
3. **LoanAmount_Log** - Loan size risk
4. **Ability_to_Pay** - Income-to-loan ratio
5. Property_Area, Married status, Education level

### Evaluation Metrics
- **Precision**: Accuracy of positive predictions
- **Recall**: Ability to catch actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual breakdown of predictions

### Threshold Impact
| Threshold | Recall (Class 0) | Interpretation |
|-----------|------------------|----------------|
| 0.4 | Lower | More approvals, higher risk |
| 0.5 | Moderate | Balanced approach |
| 0.6 | Higher | Fewer approvals, lower risk ✅ |
| 0.7 | Highest | Very conservative |

## 📊 Visualizations

### 1. Distribution Analysis
- Before/after log transformation comparison
- Skewness reduction visualization

### 2. Class Balance
- Loan status distribution charts
- SMOTE impact on training set

### 3. Feature Importance Bar Plot
- Identifies which features drive decisions

### 4. Probability Distribution
- Separation between approved and rejected applications
- Model confidence visualization

### 5. Error Map
- Scatter plot showing correct vs incorrect predictions
- Income vs Loan Amount relationship

### 6. Decision Boundaries
- 2D visualization of model decisions
- Separate plots for good vs bad credit history
- Shows how income and loan amount interact

## 🚀 How to Use

### Installation
```bash
# Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn joblib
```

### Running the Notebook
1. Place `loan-train.csv` in the same directory as the notebook
2. Update the file path in the first cell if needed
3. Run all cells sequentially

### Making Predictions
```python
# Load the saved model
import joblib
model = joblib.load('loan_voting_model.pkl')

# Make a prediction
status, confidence = predict_loan_status(
    gender='Male',
    married='Yes',
    dependents=0,
    education='Graduate',
    self_employed='No',
    applicant_income=6000,
    coapplicant_income=2000,
    loan_amount=120,
    loan_term=360,
    credit_history=1.0,
    property_area='Semiurban'
)
print(f"Decision: {status} (Confidence: {confidence:.2%})")
```

## 🎓 Key Learnings

1. **SMOTE is Critical**: Dramatically improves minority class prediction
2. **Feature Engineering Matters**: Derived features (Ability_to_Pay) boost performance
3. **Threshold Tuning**: Moving from 0.5 to 0.6 reduces risk exposure
4. **Ensemble Power**: Voting/Stacking outperforms individual models
5. **Log Transforms**: Essential for handling skewed financial data
6. **Credit History Dominates**: Single most important predictor

## 📁 Project Structure
```
loan-approval-prediction/
│
├── modified_ML_project.ipynb    # Main notebook with complete pipeline
├── README.md                     # This file
├── loan-train.csv               # Training dataset (not included)
├── loan_voting_model.pkl        # Saved model (generated after running)
└── requirements.txt             # Python dependencies
```

## 🔮 Future Improvements
- [ ] Deploy as web application (Flask/Streamlit)
- [ ] Add explainability (SHAP/LIME)
- [ ] Incorporate external credit score data
- [ ] Implement real-time prediction API
- [ ] A/B testing framework for threshold optimization
- [ ] Add time-series analysis for income stability
- [ ] Cross-validation with multiple datasets

## 📝 Model Interpretation

### Why Voting Classifier?
- **Diverse Perspectives**: RF captures feature interactions, XGBoost captures gradients
- **Reduced Overfitting**: Averaging smooths out individual model biases
- **Improved Generalization**: Better performance on unseen data

### Why Threshold = 0.6?
- **Risk Management**: Banks prefer false negatives (reject good) over false positives (approve bad)
- **Regulatory Compliance**: Reduces default rates
- **Business Strategy**: Conservative approach protects capital

## 🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for improvements.

## 📄 License
This project is available for educational purposes.

## 👤 Author
**Shreyank Jaiswal**

## 🙏 Acknowledgments
- Dataset source: Loan prediction practice problem
- Inspired by real-world banking risk assessment systems
- Community contributions from scikit-learn and XGBoost teams

---

**Note**: This is a learning project. For production use, additional validation, monitoring, and compliance checks are required.
