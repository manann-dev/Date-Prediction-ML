import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

df['RepurchaseWithin1Month'] = df.groupby('PartyName')['InvoiceDate'].diff().dt.days.lt(30).astype(int)
df['RepurchaseWithin1Month'] = df['RepurchaseWithin1Month'].fillna(0).astype(int)

df['TimeSinceLastPurchase'] = df.groupby('PartyName')['InvoiceDate'].diff().dt.days.fillna(0)
df['AveragePurchaseAmount'] = df.groupby('PartyName')['Amount'].transform('mean')
df['TotalNumberOfPurchases'] = df.groupby('PartyName').cumcount() + 1

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

features = ['TimeSinceLastPurchase', 'AveragePurchaseAmount', 'TotalNumberOfPurchases']
X_train = train_df[features]
y_train = train_df['RepurchaseWithin1Month']
X_test = test_df[features]
y_test = test_df['RepurchaseWithin1Month']

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}\n')
print('Confusion Matrix:')
print(conf_matrix)
print('\nClassification Report:')
print(classification_rep)
