import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import streamlit as st

# Load data
data = pd.read_csv('Stock_Data.csv') 

# Sidebar  
st.sidebar.header('Options')
stock = st.sidebar.selectbox('Select stock', data.columns[1:-1]) 

option = st.sidebar.radio('Analysis or Prediction', ['Analysis', 'Prediction'])

# Analysis
if option == 'Analysis':
    
    all_time = st.sidebar.checkbox('All Time') 

    if all_time:
        periods = len(data)
    else:
        periods = st.sidebar.slider('Select period (in days)', min_value=30, max_value=365, value=90)
        
    data = data.iloc[-periods:] 

    ma50 = st.sidebar.checkbox('50 Day Moving Average')
    ma200 = st.sidebar.checkbox('200 Day Moving Average')  
    rsi = st.sidebar.checkbox('RSI (14 days)')
    bbands = st.sidebar.checkbox('Bollinger Bands (20 days)')

    # Calculate indicators
    if ma50:
        data[f'MA50_{stock}'] = data[stock].rolling(50).mean() 

    if ma200:
        data[f'MA200_{stock}'] = data[stock].rolling(200).mean()
        
    if bbands:   
        data[f'STD_{stock}'] = data[stock].rolling(20).std()
        data[f'MA50_{stock}'] = data[stock].rolling(50).mean() 
        data[f'Upper_Band_{stock}'] = data[f'MA50_{stock}'] + (2 * data[f'STD_{stock}'])
        data[f'Lower_Band_{stock}'] = data[f'MA50_{stock}'] - (2 * data[f'STD_{stock}'])
        
    if rsi:
        delta = data[stock].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean() 
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data[f'RSI_{stock}'] = 100 - (100 / (1 + rs))
        
    # Create figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data['Date'], y=data[stock], mode='lines', name=f'Close {stock}'))

    if ma50:
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'MA50_{stock}'], mode='lines', name=f'MA50 {stock}'))
        
    if ma200:
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'MA200_{stock}'], mode='lines', name=f'MA200 {stock}'))
        
    if rsi:
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'RSI_{stock}'], mode='lines', name=f'RSI {stock}'))
        
    if bbands:
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'Upper_Band_{stock}'], fill='tonexty', mode='lines', name=f'Upper Band {stock}', fillcolor='rgba(0,100,80,0.2)'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data[f'Lower_Band_{stock}'], fill='tonexty', mode='lines', name=f'Lower Band {stock}', fillcolor='rgba(0,100,80,0.2)'))

    fig.update_layout(title=f'{stock} Technical Indicators') 

    st.plotly_chart(fig)
    
# Prediction    
if option == 'Prediction':
    
    X = data.index.values.reshape(-1, 1)
    y = data[stock].values
    lr = LinearRegression()
    lr.fit(X, y)

    # Align X predictions to index
    start = data.index.max() + 1
    end = start + 365
    prediction_dates = pd.date_range(start=data['Date'].iloc[-1], periods=365)
    X_pred = np.arange(start, end).reshape(-1, 1)
    lr_preds = lr.predict(X_pred)

    # ARIMA
    model = ARIMA(data[stock], order=(5, 1, 0))
    fitted_model = model.fit()
    forecast = fitted_model.forecast(steps=365)
    arima_preds = forecast.values

    # Create figure for prediction
    prediction_fig = go.Figure()

    prediction_fig.add_trace(go.Scatter(x=data['Date'], y=data[stock], mode='lines', name=f'Close {stock}'))
    prediction_fig.add_trace(
        go.Scatter(x=prediction_dates, y=lr_preds, mode='lines', name=f'Linear Regression {stock}'))
    prediction_fig.add_trace(
        go.Scatter(x=prediction_dates, y=arima_preds, mode='lines', name=f'ARIMA {stock}'))

    prediction_fig.update_layout(title=f'{stock} Price Prediction')
    
    # Customize x-axis labels to show the full date
    prediction_fig.update_xaxes(
        tickformat='%Y-%m-%d',
        #dtick='M1',  # Show monthly ticks
        #ticklabelmode="period")
    )

    st.plotly_chart(prediction_fig)
