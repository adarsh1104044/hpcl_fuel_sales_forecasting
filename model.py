import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def preprocess_single_fuel_excel(file, fuel_type):
    """
    Converts a wide-format Excel file to long format correctly
    """
    df = pd.read_excel(file)
    
    # Get the first two columns (Ship To, Outlet)
    id_cols = df.columns[:2].tolist()
    
    # Get all month columns (everything after the first two)
    value_cols = df.columns[2:].tolist()
    
    # Melt the dataframe
    melted = df.melt(
        id_vars=id_cols, 
        value_vars=value_cols,
        var_name='Month', 
        value_name='Sales'
    )
    
    # Add the fuel type based on which file this is
    melted['Fuel_Type'] = fuel_type
    
    # Parse months to datetime
    melted['Month'] = pd.to_datetime(melted['Month'], errors='coerce')
    
    # Remove rows with missing data
    melted = melted.dropna(subset=['Month', 'Sales'])
    
    # Rename columns to match expected format
    melted.columns = ['Ship To', 'Outlet', 'Month', 'Sales', 'Fuel_Type']
    
    return melted

def prepare_time_series(melted_df, outlet, fuel_type):
    filtered = melted_df[(melted_df['Outlet'] == outlet) & (melted_df['Fuel_Type'] == fuel_type)]
    filtered = filtered[['Month', 'Sales']].rename(columns={'Month': 'ds', 'Sales': 'y'})
    filtered = filtered.dropna().sort_values('ds')
    return filtered

def forecast_sales(ts_df, periods):
    model = Prophet()
    model.fit(ts_df)
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    
    # Separate historical and future predictions
    historical_forecast = forecast[forecast['ds'] <= ts_df['ds'].max()]
    future_forecast = forecast[forecast['ds'] > ts_df['ds'].max()]
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], ts_df, historical_forecast, future_forecast

def plot_actual_vs_predicted(ts_df, forecast, historical_forecast, future_forecast, periods):
    """
    Plot actual vs predicted values with confidence intervals
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot actual values
    ax.plot(ts_df['ds'], ts_df['y'], 'bo-', label='Actual Sales', linewidth=2, markersize=6)
    
    # Plot historical predictions (fitted values)
    ax.plot(historical_forecast['ds'], historical_forecast['yhat'], 'g--', 
            label='Historical Predictions', linewidth=2, alpha=0.7)
    
    # Plot future predictions
    ax.plot(future_forecast['ds'], future_forecast['yhat'], 'r-', 
            label='Future Forecast', linewidth=2)
    
    # Plot confidence intervals for future predictions only
    ax.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], 
                    future_forecast['yhat_upper'], color='orange', alpha=0.3, 
                    label='Forecast Confidence Interval')
    
    # Add vertical line to separate historical and forecasted data
    split_date = ts_df['ds'].max()
    ax.axvline(x=split_date, color='gray', linestyle='--', alpha=0.7, 
               label='Forecast Start', linewidth=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.set_title('Actual vs Predicted Sales with Future Forecast')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig
