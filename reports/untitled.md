# Model Building and Selection Report

## Objective

The objective of this phase was to train and evaluate multiple machine learning regression models for predicting house prices and identify the most suitable model for deployment.

---

# Feature Preparation

After data cleaning and preprocessing, the final dataset contained:

- 7,296 records
- Numerical features:
  - Total Square Feet
  - Bathrooms
  - Balconies
  - BHK

- Categorical features:
  - Area Type
  - Location

---

# Encoding Categorical Variables

Machine learning models cannot directly process categorical values.

Therefore, One-Hot Encoding was applied to:

- Area Type
- Location

### Encoding Summary

| Feature | Categories |
|----------|-----------|
| Area Type | 4 |
| Location | 224 |

Using `drop_first=True`:

- Area Type generated 3 dummy columns
- Location generated 223 dummy columns

Total feature count after encoding:

```text
230 features
```

---

# Train-Test Split

The dataset was divided into:

- Training Set: 80%
- Testing Set: 20%

Configuration:

```python
test_size = 0.20
random_state = 42
```

Dataset Sizes:

| Dataset | Shape |
|----------|----------|
| X Train | (5836, 230) |
| X Test | (1460, 230) |

---

# Evaluation Metrics

The following regression metrics were used:

## R² Score

Measures how much variance in house prices is explained by the model.

Higher values indicate better performance.

---

## MAE (Mean Absolute Error)

Average prediction error.

Lower values indicate better performance.

---

## RMSE (Root Mean Squared Error)

Penalizes larger prediction errors more heavily.

Lower values indicate better performance.

---

# Models Evaluated

## 1. Linear Regression

Linear Regression was used as the baseline model.

### Results

| Metric | Value |
|----------|----------|
| MAE | 18.52 |
| RMSE | 35.72 |
| R² | 0.8411 |

Interpretation:

- Predictions are off by approximately ₹18.5 Lakhs on average.
- The model explains approximately 84.1% of house price variation.

---

## 2. Ridge Regression

Ridge Regression introduces L2 regularization to reduce overfitting.

### Results

| Metric | Value |
|----------|----------|
| R² | 0.8370 |

Observation:

Performance was slightly lower than Linear Regression.

---

## 3. Lasso Regression

Lasso Regression applies L1 regularization and can eliminate less important features.

### Results

| Metric | Value |
|----------|----------|
| R² | 0.8246 |

Observation:

Performance decreased compared to Linear Regression and Ridge Regression.

---

## 4. Decision Tree Regressor

Decision Trees can capture nonlinear relationships between features and house prices.

### Results

| Metric | Value |
|----------|----------|
| R² | 0.8389 |

Observation:

Performance was comparable to Linear Regression but slightly lower.

---

## 5. Random Forest Regressor

Random Forest combines multiple decision trees to improve prediction accuracy.

### Results

| Metric | Value |
|----------|----------|
| R² | 0.8550 |

Observation:

Random Forest achieved the highest test-set R² score among all models.

---

# Cross Validation

A single train-test split may sometimes provide optimistic results.

To evaluate model stability, ShuffleSplit Cross Validation was performed.

Configuration:

```python
n_splits = 5
test_size = 0.20
random_state = 42
```

---

## Linear Regression

Cross Validation Scores:

```text
0.84+
```

Average R²:

```text
0.8467
```

---

## Random Forest

Cross Validation Scores:

```text
[0.8550, 0.8344, 0.7785, 0.8151, 0.8629]
```

Average R²:

```text
0.8292
```

---

# Model Comparison

| Model | Test R² |
|---------|---------|
| Linear Regression | 0.8411 |
| Ridge Regression | 0.8370 |
| Lasso Regression | 0.8246 |
| Decision Tree | 0.8389 |
| Random Forest | 0.8550 |

---

# Final Model Selection

Although Random Forest achieved the highest score on the test set, Linear Regression demonstrated stronger and more consistent performance during cross-validation.

### Selected Model

**Linear Regression**

Reasons:

- Strong predictive performance
- Highest cross-validation score
- Simpler architecture
- Faster training and inference
- Better interpretability
- Easier deployment for API and chatbot integration

---

# Prediction Function

A custom prediction function was implemented to:

1. Accept user inputs:
   - Location
   - Area Type
   - Total Square Feet
   - Bathrooms
   - Balconies
   - BHK

2. Apply the same feature encoding used during training.

3. Generate house price predictions using the trained Linear Regression model.

---

# Model Export

The trained model was exported using Pickle.

Saved files:

```text
Models/
├── house_price_model.pkl
├── model_columns.json
```

These files will be used during deployment and GenAI chatbot integration.

---

# Conclusion

Five regression models were trained and evaluated for house price prediction.

Among them, Linear Regression provided the best balance of:

- Accuracy
- Generalization
- Interpretability
- Deployment simplicity

The final model achieved a cross-validated R² score of approximately **0.85**, indicating strong predictive capability on unseen data and making it suitable for integration into the next phase of the project: the GenAI-powered house recommendation chatbot.