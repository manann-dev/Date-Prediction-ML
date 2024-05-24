import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model_filename = 'barge.joblib'
loaded_model = joblib.load(model_filename)
print("Model loaded successfully")

user_input_data = {'PartyName': [], 'InvoiceDate': [], 'Amount': []}
new_data = pd.DataFrame()

while True:
    party_name = input("Enter the party name: ")

    for i in range(5):
        date_str = input(f"Enter the date {i + 1} in YYYY-MM-DD format: ")
        amount = float(input(f"Enter the amount for date {i + 1}: "))

        user_input_data['PartyName'].append(party_name)
        user_input_data['InvoiceDate'].append(datetime.strptime(date_str, "%Y-%m-%d"))
        user_input_data['Amount'].append(amount)

    user_input_df = pd.DataFrame(user_input_data)
    new_data = pd.concat([new_data, user_input_df], ignore_index=True)

    add_more_dates = input("Do you want to add more dates for this party? (yes/no): ").lower()
    
    if add_more_dates != 'yes':
        add_another_party = input("Do you want to add another party? (yes/no): ").lower()

        if add_another_party != 'yes':
            break 

        user_input_data = {'PartyName': [], 'InvoiceDate': [], 'Amount': []}

new_data['TimeSinceLastPurchase'] = new_data.groupby('PartyName')['InvoiceDate'].diff().dt.days.fillna(0)
new_data['AveragePurchaseAmount'] = new_data.groupby('PartyName')['Amount'].transform('mean')
new_data['TotalNumberOfPurchases'] = new_data.groupby('PartyName').cumcount() + 1

imputer = SimpleImputer(strategy='mean')
new_data_imputed = pd.DataFrame(imputer.fit_transform(new_data[['TimeSinceLastPurchase', 'AveragePurchaseAmount', 'TotalNumberOfPurchases']]),
                                columns=['TimeSinceLastPurchase', 'AveragePurchaseAmount', 'TotalNumberOfPurchases'])
new_data[['TimeSinceLastPurchase', 'AveragePurchaseAmount', 'TotalNumberOfPurchases']] = new_data_imputed

features = ['TimeSinceLastPurchase', 'AveragePurchaseAmount', 'TotalNumberOfPurchases']
X_new = new_data[features] 
y_pred_new = loaded_model.predict(X_new)

new_data['PredictedTimeUntilNextPurchase'] = y_pred_new

threshold = 10

new_data['PredictedPurchaseBinary'] = (y_pred_new <= threshold).astype(int)

print(new_data[['PartyName', 'InvoiceDate', 'PredictedTimeUntilNextPurchase', 'PredictedPurchaseBinary']])

max_predicted_date = new_data['InvoiceDate'].max() + pd.to_timedelta(y_pred_new.max(), unit='D')
new_data['PredictedDate'] = max_predicted_date
true_labels = (new_data['PredictedTimeUntilNextPurchase'] <= threshold).astype(int)

accuracy = accuracy_score(true_labels, new_data['PredictedPurchaseBinary'])
print(f'Accuracy Score: {accuracy}')

conf_matrix = confusion_matrix(true_labels, new_data['PredictedPurchaseBinary'])
print(f'Confusion Matrix:\n{conf_matrix}')

class_report = classification_report(true_labels, new_data['PredictedPurchaseBinary'])
print(f'Classification Report:\n{class_report}')

print(new_data[['PartyName', 'InvoiceDate', 'PredictedTimeUntilNextPurchase', 'PredictedDate']])