import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import joblib


np.random.seed(42)

num_rows = 1000

parties = ['ABC Shipping', 'XYZ Boats', 'LMN Maritime', 'PQR Logistics']
party_data = np.random.choice(parties, num_rows)

start_date = datetime.now() - timedelta(days=365)
end_date = datetime.now()
invoice_dates = [np.random.choice(pd.date_range(start_date, end_date)) for _ in range(num_rows)]


amounts = np.random.uniform(low=100, high=1000, size=num_rows)


data = {
    'InvoiceID': range(1, num_rows + 1),
    'PartyName': party_data,
    'InvoiceDate': invoice_dates,
    'Amount': amounts
}

df = pd.DataFrame(data)


df = df.sort_values(by=['PartyName', 'InvoiceDate'])

df['TimeUntilNextPurchase'] = df.groupby('PartyName')['InvoiceDate'].diff().dt.days


df['TimeSinceLastPurchase'] = df.groupby('PartyName')['InvoiceDate'].diff().dt.days.fillna(0)
df['AveragePurchaseAmount'] = df.groupby('PartyName')['Amount'].transform('mean')
df['TotalNumberOfPurchases'] = df.groupby('PartyName').cumcount() + 1


df = df.dropna(subset=['TimeUntilNextPurchase'])


current_date_data = {
    'PartyName': df['PartyName'].unique(),
    'InvoiceDate': datetime.now(),
    'Amount': 0  
}
current_date_df = pd.DataFrame(current_date_data)

df = pd.concat([df, current_date_df], ignore_index=True)

imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df[['TimeSinceLastPurchase', 'AveragePurchaseAmount', 'TotalNumberOfPurchases']]),
                           columns=['TimeSinceLastPurchase', 'AveragePurchaseAmount', 'TotalNumberOfPurchases'])
df[['TimeSinceLastPurchase', 'AveragePurchaseAmount', 'TotalNumberOfPurchases']] = df_imputed


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df = train_df.dropna(subset=['TimeUntilNextPurchase'])


features = ['TimeSinceLastPurchase', 'AveragePurchaseAmount', 'TotalNumberOfPurchases']
X_train = train_df[features]
y_train = train_df['TimeUntilNextPurchase']
X_test = test_df[features]
y_test = test_df['TimeUntilNextPurchase']

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

model_filename = 'barge.joblib'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

loaded_model = joblib.load(model_filename)
print("Model loaded successfully")

y_pred = loaded_model.predict(X_test)

predicted_dates = test_df[['PartyName', 'InvoiceDate']].copy()
predicted_dates['PredictedDate'] = predicted_dates.groupby('PartyName')['InvoiceDate'].transform('max') + pd.to_timedelta(y_pred, unit='D')

latest_predictions = predicted_dates.groupby('PartyName')['PredictedDate'].max().reset_index()
latest_predictions.to_csv('latest_predictions1.csv', index=False)

print(latest_predictions)
