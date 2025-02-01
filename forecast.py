import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate

def read_and_predict_sales(csv_file_path):
    try:
        # Read CSV file
        historical_data = pd.read_csv(csv_file_path)
        
        # Verify columns exist
        required_columns = ['date', 'sales']
        if not all(col in historical_data.columns for col in required_columns):
            raise ValueError("CSV must contain 'date' and 'sales' columns")
        
        # Convert dates to datetime
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        
        # Sort by date to ensure correct order
        historical_data = historical_data.sort_values('date').reset_index(drop=True)
        
        # Verify we have at least 12 months of data for sufficient samples
        if len(historical_data) < 12:
            raise ValueError("Need at least 12 months of historical data")
        
        # Take the last 12 months if we have more data
        historical_data = historical_data.tail(12)
        
        # Calculate month-over-month sales difference
        historical_data['sales_diff'] = historical_data['sales'].diff()
        historical_data = historical_data.dropna()
        
        # Create supervised dataset with 2 lagged features
        feature_names = [f'month_{i}' for i in range(1, 3)]  # Use 2 lags
        supervised_data = pd.DataFrame()
        for i, name in enumerate(feature_names, 1):
            supervised_data[name] = historical_data['sales_diff'].shift(i)
        supervised_data['sales_diff'] = historical_data['sales_diff']
        supervised_data = supervised_data.dropna().reset_index(drop=True)
        
        # Prepare data for training (scale features only)
        X = supervised_data[feature_names].values
        y = supervised_data['sales_diff'].values
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
        X_scaled = scaler.transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Prepare prediction data (last 2 differences)
        last_differences = historical_data['sales_diff'].tail(2).values
        
        # Generate 6 months of predictions
        predictions = []
        current_differences = last_differences.copy()
        last_actual_sale = historical_data['sales'].iloc[-1]
        
        for _ in range(6):
            current_input = current_differences.reshape(1, -1)
            scaled_input = scaler.transform(current_input)
            next_diff = model.predict(scaled_input)[0]
            predictions.append(next_diff)
            # Update differences: remove oldest, add new prediction
            current_differences = np.roll(current_differences, -1)
            current_differences[-1] = next_diff
        
        # Calculate actual sales values from differences
        future_sales = []
        current_sale = last_actual_sale
        for diff in predictions:
            current_sale += diff
            future_sales.append(current_sale)
        
        # Create future dates
        last_date = historical_data['date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='M')
        
        # Create prediction DataFrame
        predictions_df = pd.DataFrame({
            'date': future_dates,
            'predicted_sales': [round(sale, 3) for sale in future_sales]
        })
        
        # Create combined table view
        historical_table = historical_data[['date', 'sales']].copy()
        historical_table['type'] = 'Historical'
        historical_table = historical_table.rename(columns={'sales': 'value'})
        
        predictions_table = predictions_df.copy()
        predictions_table['type'] = 'Predicted'
        predictions_table = predictions_table.rename(columns={'predicted_sales': 'value'})
        
        # Display tables
        # print("\n=== Sales Analysis Report ===\n")
        
        # print("Historical Sales:")
        # print(tabulate(historical_table, headers='keys', tablefmt='pretty', showindex=False))
        
        # print("\nPredicted Sales:")
        # print(tabulate(predictions_table, headers='keys', tablefmt='pretty', showindex=False))
        
        return historical_data, predictions_df
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None, None