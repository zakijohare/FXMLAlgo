import os
import json
import MetaTrader5 as mt5
import pandas as pd

def MT5Connect():
    # Determine the directory of the current script
    script_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    credentials_file = os.path.join(script_dir, 'credentials.json')
    
    # Load credentials from JSON file
    with open(credentials_file, 'r') as file:
        credentials = json.load(file)
    
    server = credentials['server']
    login = int(credentials['login'])
    password = credentials['password']
    
    # Initialize MT5 and log in using the credentials
    mt5.initialize()
    return mt5.login(login, password=password, server=server)

def GetRates(symbol, n, timeframe):
    if not MT5Connect():
        return pd.DataFrame()  # Return empty DataFrame if login fails
    
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    mt5.shutdown()

    # Convert rates to DataFrame and format time
    rates_frame = pd.DataFrame(rates)
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame.set_index('time', inplace=True)
    
    # Drop unnecessary columns
    rates_frame.drop(columns=['tick_volume', 'spread', 'real_volume'], inplace=True)

    return rates_frame
