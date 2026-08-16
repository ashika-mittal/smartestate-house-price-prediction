Linear Regression Results

After data cleaning, feature engineering, outlier removal, and one-hot encoding, a Linear Regression model was trained using an 80-20 train-test split.

Results:

* R² Score: 0.8411
* MAE: 18.52 lakhs
* RMSE: 35.72 lakhs

Observations:

* The model explains approximately 84% of the variance in house prices.
* Total square footage and location were found to be strong predictors.
* Outlier removal significantly improved model performance.
* One-hot encoding of location allowed the model to capture locality-specific pricing patterns.

Linear Regression- * R² Score: 0.8411
Ridge- * R² Score: 0.8370

Observation:
Linear Regression achieved the highest R² score and outperformed Ridge Regression on the test set. Since the dataset was already cleaned and outliers were removed, additional regularization did not improve performance.

Lasso-
 R² Score: 0.82461483996254

 Interpretation
* Linear Regression performed best.
* Ridge was slightly worse.
* Lasso was noticeably worse.



This tells us:

 The relationship between features and house price is largely linear.
 Your cleaning and outlier removal worked well.
 Heavy regularization is not needed.
 There isn’t much overfitting for Ridge/Lasso to fix.
 Lasso is shrinking some useful coefficients too aggressively.


Conclusion:

Linear Regression achieved the highest predictive performance with an R² score of 84.11%, indicating that the model explains approximately 84% of the variation in house prices.

Although Ridge and Lasso Regression help reduce overfitting through regularization, they did not improve performance on the cleaned dataset. Therefore, Linear Regression was selected as the final model for house price prediction.
