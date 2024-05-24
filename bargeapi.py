import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import json
from datetime import datetime
from flask import Flask, jsonify

app = Flask(__name__)

def predict_next_purchase(df, current_date):
    df['DaysSinceLastPurchase'] = df.index.to_series().diff().dt.days
    df['PurchaseAmount7DaysAgo'] = df['Amount'].shift(1)
    df['StationaryAmount'] = df['Amount'].diff()

    train_size = int(0.8 * len(df))
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 7)

    model = SARIMAX(train['DaysSinceLastPurchase'].dropna(), order=order, seasonal_order=seasonal_order)
    result = model.fit()

    forecast_days = 7
    forecast_dates = pd.date_range(start=test.index[-1] + timedelta(days=1), periods=forecast_days, freq='D')
    forecast_index = pd.DatetimeIndex(forecast_dates)

    forecast_series = pd.Series(result.forecast(steps=forecast_days), index=forecast_index)

    results_dict = {}

    for party in df['PartyName'].unique():
        party_data = df[df['PartyName'] == party]
        party_forecast = result.get_forecast(steps=len(party_data) + forecast_days)

        last_index = pd.to_datetime(party_forecast.predicted_mean.index[-1])
        forecast_index = pd.date_range(start=current_date, periods=forecast_days, freq='D')
        probabilities = party_forecast.predicted_mean.iloc[-forecast_days:]

        probabilities = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())

        next_purchase_dates = [date.strftime('%Y-%m-%d') for date in forecast_index]

        results_dict[party] = {
            'NextPurchaseDates': dict(zip(next_purchase_dates, probabilities.to_dict().values())),
        }
    
    return results_dict

@app.route('/predict', methods=['GET'])
def get_next_purchase_predictions():
    df = pd.read_csv('testdata1.csv')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df.set_index('InvoiceDate')
    df = df.sort_index()

    current_date = datetime.now().date()
    results = predict_next_purchase(df, current_date)

    json_results = json.dumps(results, indent=2)
    return jsonify(json_results)

if __name__ == '__main__':
    app.run(debug=True)
