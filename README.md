# Prediction-of-Adult-Income-Based-on-Census-Data
This project aims to predict whether an individual's income exceeds $50K per year based on demographic and economic features from the U.S. Census dataset. Using Machine Learning models, the study evaluates how factors like age, education, occupation, and marital status impact income levels.

Project Goals
1. Classify individuals into two income categories: ≤ $50K and > $50K
2. Analyze key socio-economic factors affecting income levels
3. Compare different Machine Learning models for best performance

Dataset
Source: UCI Machine Learning Repository – Adult Census Income Dataset

Features:
1. Demographics: Age, Gender, Race, Marital Status
2. Education: Highest degree attained
3. Employment: Occupation, Workclass, Hours-per-week
4. Socioeconomic: Capital gain, Capital loss, Relationship
5. Target Variable: Income Category (≤ $50K or > $50K per year)

Technologies Used
1. Python – Data processing & model training
2. Pandas & NumPy – Data cleaning and transformation
3. Scikit-Learn – Machine Learning models
4. Matplotlib & Seaborn – Data visualization
5. Feature Engineering – Handling categorical & numerical data
6. Hyperparameter Tuning – Optimizing model performance

Machine Learning Models Implemented
1. Logistic Regression – Baseline model for classification
2. Decision Tree Classifier – Capturing non-linear relationships
3. Random Forest Classifier – Improving accuracy with ensemble learning
4. Gradient Boosting (XGBoost) – Advanced boosting for better performance
5. Support Vector Machine (SVM) – Evaluating feature separation
6. Neural Networks – Exploring deep learning approaches

Key Insights & Findings
1. Education & Occupation significantly impact income levels
2. Workclass & Hours Worked show a strong correlation with income
3. Ensemble models (Random Forest & XGBoost) outperform baseline models
4. Feature importance analysis highlights the most influential predictors

Performance Metrics
1. Accuracy, Precision, Recall, and F1-score evaluated for each model
2. ROC-AUC Curve for assessing classifier performance
3. Cross-validation to ensure model generalization

Impact & Applications
1. Data-driven insights for policymakers and economists on income disparities
2. HR & Workforce Planning – Identifying factors influencing career success
3. Predictive analytics for salary benchmarking in job markets
