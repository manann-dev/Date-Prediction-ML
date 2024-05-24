# Barge Predictive Model

This project involves building a predictive model to estimate the time until the next purchase for different parties involved in shipping and logistics.

## Overview

The dataset consists of synthetic data generated for party names, invoice dates, and amounts. The goal is to predict the time until the next purchase for each party based on various features, including the time since the last purchase, average purchase amount, and total number of purchases.

## Files

- `train_model.py`: The Python script containing the code for data generation, feature engineering, model training, and prediction.
- `barge.joblib`: The saved RandomForestRegressor model using joblib.
- `latest_predictions1.csv`: CSV file containing the latest predictions for each party.
- 'accuracy_test.py' : For the accuracy, R1-Score.

## Dependencies

- pandas
- numpy
- scikit-learn
- joblib

## How to Run

1. Ensure you have the required dependencies installed (`pip install pandas numpy scikit-learn joblib`).
2. Run the `train_model.py` script to generate data, train the model, and make predictions.
3. Check the console for model-related messages and the `latest_predictions1.csv` file for the latest predictions.

## Notes

- The script uses RandomForestRegressor from scikit-learn for regression.
- Features include time since the last purchase, average purchase amount, and total number of purchases.
- The trained model is saved as `barge.joblib`.
- Predictions are stored in the `latest_predictions1.csv` file.

## Author

Manan Raval

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
