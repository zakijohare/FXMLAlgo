import json
import os
import MetaTrader5 as mt5
import pandas as pd
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ta import add_all_ta_features
from ta.utils import dropna
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials, space_eval
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from copy import deepcopy as dc
import requests
from datetime import datetime, timedelta


def FetchJSONNews(url):
    # Fetch data from the URL
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    events = response.json()  # Return the JSON data directly

    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=['Event Time'])

    for event in events:
        # Check if the event is a high-impact USD event
        if event['country'] == 'USD' and event['impact'] == 'High':
            # Parse the event datetime (remove the 'Z' and convert to naive datetime)
            event_time = datetime.fromisoformat(event['date'].replace('Z', '+00:00')).replace(tzinfo=None)
            # Add 7 hours to the event time
            adjusted_time = event_time + timedelta(hours=7)
            # Format the datetime to remove any timezone information
            adjusted_time_str = adjusted_time.strftime("%Y-%m-%d %H:%M:%S")
            # Create a DataFrame for the new row and concatenate it
            new_row = pd.DataFrame({'Event Time': [adjusted_time_str]})
            df = pd.concat([df, new_row], ignore_index=True)
    
    return df

def MT5Connect():
    # Determine the directory of the currently executing script
    script_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
    
    # Construct the path to the credentials file at the same level as the script directory
    credentials_file = os.path.join(script_dir, 'credentials.json')
    
    # Load the credentials from the JSON file
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
        return pd.DataFrame()
    
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    mt5.shutdown()

    rates_frame = pd.DataFrame(rates)
    rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
    rates_frame.set_index('time', inplace=True)
    
    # Remove unwanted columns
    rates_frame.drop(columns=['tick_volume', 'spread', 'real_volume'], inplace=True)

    return rates_frame

def IdentifyNonStatinary(x):
 
    non_stationaries = []
    for col in x.columns:
        # Perform Augmented Dickey-Fuller test only on numeric columns
        if x[col].dtype == 'float64' or x[col].dtype == 'int64':
            result = adfuller(x[col].dropna())  # Drop NA values as ADF doesn't handle them
            p_value = result[1]
            test_statistic = result[0]
            critical_value = result[4]["1%"]

            # Check if p-value is above 0.05 or test statistic is higher than critical value
            if p_value > 0.05 or test_statistic > critical_value:
                non_stationaries.append(col)

    # Optionally print the number of non-stationary features found
    print(f"Non-Stationary Features Found: {len(non_stationaries)}")

    return non_stationaries

def StationaryAndScale(df, non_stationaries):
    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Make the non-stationary columns stationary using pct_change
    df[non_stationaries] = df[non_stationaries].pct_change()

    # Drop the first row as it will be NaN after pct_change
    df = df.iloc[1:].copy()  # Ensure we are working with a fresh copy

    # Find and drop columns with NaN values
    na_list = df.columns[df.isna().any()].tolist()
    df.drop(columns=na_list, inplace=True)

    # Handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)  # Ensure no NaNs are left after replacing infinities

    # Feature Scaling
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # Convert the numpy array returned by StandardScaler back to a DataFrame
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

    # Return the scaled DataFrame and the scaler
    return df_scaled, scaler

def LiveStationaryAndScale(df, non_stationaries, scaler):
    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Make the non-stationary columns stationary using pct_change
    df[non_stationaries] = df[non_stationaries].pct_change()

    # Drop the first row as it will be NaN after pct_change
    df = df.iloc[1:].copy()  # Ensure we are working with a fresh copy

    # Find and drop columns with NaN values
    na_list = df.columns[df.isna().any()].tolist()
    df.drop(columns=na_list, inplace=True)

    # Handle infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)  # Ensure no NaNs are left after replacing infinities

    # Use the provided scaler to transform the DataFrame
    df_scaled = scaler.transform(df)

    # Convert the numpy array returned by StandardScaler back to a DataFrame
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

    # Return the scaled DataFrame
    return df_scaled

def TargetPCT(df):
    # Calculate the absolute percentage change
    df['abs_pct_change'] = df['close'].pct_change().abs()
    df = df.iloc[1:].copy()  # Drop the first row as it will be NaN after pct_change and make a copy to avoid chained assignment issues

    # Identify outliers using IQR in the training set
    Q1 = df['abs_pct_change'].quantile(0.25)
    Q3 = df['abs_pct_change'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Create a mask for non-outlier data in the training set
    non_outliers = (df['abs_pct_change'] >= lower_bound) & (df['abs_pct_change'] <= upper_bound)

    # Calculate the mean of the absolute percentage changes excluding outliers in the training set
    mean_pct_change = df.loc[non_outliers, 'abs_pct_change'].mean()

    # Define labels based on the percentage change relative to the mean for the entire dataset
    df['signal'] = (df['abs_pct_change'] > mean_pct_change).astype(int)

    # Drop the absolute percentage change column
    df.drop(columns=['abs_pct_change'], inplace=True)

    # Drop the 'low', 'open', and 'high' columns
    df.drop(columns=['low', 'open', 'high'], inplace=True)

    return df, mean_pct_change

def TargetSet(df):
   
    # Set initial signal based on whether the next close is higher or lower than the current close
    df["signal"] = 1  # Default to sell
    df.loc[df["close"].shift(-1) > df["close"], "signal"] = 2  # Change to buy if next close is higher

    # For sell positions, if the distance to next high is greater than the distance to next close, set signal to 0
    df.loc[(df["signal"] == 1) &
           (abs(df["high"].shift(-1) - df["close"]) > abs(df["close"].shift(-1) - df["close"])),
           "signal"] = 0

    # For buy positions, if the distance to next low is greater than the distance to next close, set signal to 0
    df.loc[(df["signal"] == 2) &
           (abs(df["low"].shift(-1) - df["close"]) > abs(df["close"].shift(-1) - df["close"])),
           "signal"] = 0

    # Remove rows with NaN values resulting from the shift operation
    df.dropna(inplace=True)
    
    return df

def TargetSplit(df):
    # Select all columns except the last for features, starting from the second row
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    # Return the features and target variable
    return x, y

def AddTarget(df_pca, y):
    
    # Reset index if needed for both 'df_pca' and 'y' to ensure their indices match
    if not df_pca.index.equals(y.index):
        df_pca.reset_index(drop=True, inplace=True)
        y = y.reset_index(drop=True)
    
    # Add the target variable to the PCA DataFrame
    df_pca['signal'] = y
    return df_pca

def ShiftClose(df, n_steps):
    df = dc(df)

    

    # Create lagged features for 'close'
    for i in range(1, n_steps + 1):
        df[f'Close(t-{i})'] = df['close'].shift(i)

    # Drop rows with NaN values introduced by the shift operation
    df.dropna(inplace=True)

    return df

def OptimiseAndTrain(X, y):
  
   # Constants for the optimization process
    MAX_EVALS = 15
    N_SPLITS = 5
    RANDOM_STATE = 42

    # Define the search space for hyperparameters
    space = {
        'C': hp.uniform('C', 0, 10),
        'gamma': hp.uniform('gamma', 0.001, 1),
        'kernel': hp.choice('kernel', ['linear', 'rbf']),
        'class_weight': hp.choice('class_weight', [None, 'balanced'])
    }

    # Define the objective function for optimization
    def objective(params):
        # Use 'scale' for gamma if the kernel is not 'rbf'
        gamma_value = params['gamma'] if params['kernel'] == 'rbf' else 'scale'

        # Create the SVM model with the correct hyperparameters
        svm_model = SVC(
            C=params['C'],
            kernel=params['kernel'],
            gamma=gamma_value,
            class_weight=params['class_weight'],
            random_state=RANDOM_STATE
        )

        # Setup Time Series Cross-Validator
        tscv = TimeSeriesSplit(n_splits=N_SPLITS)

        # Calculate the cross-validation score
        accuracy_scores = cross_val_score(svm_model, X, y, cv=tscv, scoring='accuracy')

        # Our goal is to maximize accuracy, so we return it as a negative value for minimization
        return {'loss': -np.mean(accuracy_scores), 'status': STATUS_OK}

    # Initialize the Trials object to keep track of results
    trials = Trials()

    # Run the optimization
    best = fmin(
    fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=MAX_EVALS,
        trials=trials,
        rstate=np.random.default_rng(RANDOM_STATE)  # For reproducibility
    )

    # Use space_eval to get the best hyperparameters with correct values
    best_hyperparams = space_eval(space, best)

    # Build and fit the best SVM model with probability estimates
    best_svm_model = SVC(
        **best_hyperparams,
        probability=True,  # Enable probability estimates
        random_state=RANDOM_STATE
    )
    best_svm_model.fit(X, y)

    return best_svm_model

def MagTrain(symbol, n, timeframe):
    df = GetRates(symbol, n, timeframe)
    df, mean_pct_change = TargetPCT(df)
    x, y = TargetSplit(df)
    non_stationaries_mag = IdentifyNonStatinary(x)
    df_scaled, MagScaler = StationaryAndScale(x, non_stationaries_mag)
    df = AddTarget(df_scaled, y)
    shifted_df = ShiftClose(df, n_steps=14)
    # Reorder the columns to move 'signal' to the end
    cols = [col for col in shifted_df.columns if col != 'signal'] + ['signal']
    shifted_df = shifted_df[cols]
    x, y = TargetSplit(shifted_df)
    MagModel = OptimiseAndTrain(x, y)
    return MagModel, mean_pct_change, MagScaler, non_stationaries_mag

def DirTrain(symbol, n, timeframe):
    df = GetRates(symbol, n, timeframe)
    df = TargetSet(df)
    x, y = TargetSplit(df)
    non_stationaries_dir = IdentifyNonStatinary(x)
    df_scaled, DirScaler = StationaryAndScale(x, non_stationaries_dir)
    df = AddTarget(df_scaled, y)
    x, y = TargetSplit(df)
    DirModel = OptimiseAndTrain(x, y)
    return DirModel, DirScaler, non_stationaries_dir

def CurrentTime():
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time


def Mag(symbol, timeframe, model, scaler, nonstat):
    df = GetRates(symbol, n=50, timeframe=timeframe)
    df = df.drop(columns=['low', 'open', 'high']) 
    df_scaled = LiveStationaryAndScale(df, nonstat, scaler)
    shifted_df = ShiftClose(df_scaled, n_steps=14)
    
    # Get the row corresponding to the current datetime
    current_time = CurrentTime()
    current_candle = shifted_df.loc[shifted_df.index == current_time]
    
    if current_candle.empty:
        print("No candle found for the current time.")
        return None
    
    PredictionMag = model.predict(current_candle)
    return PredictionMag

def Dir(symbol, timeframe, model, scaler, nonstat):
    df = GetRates(symbol, n=50, timeframe=timeframe)
    df_scaled = LiveStationaryAndScale(df, nonstat, scaler)
    
    # Get the row corresponding to the current datetime
    current_time = CurrentTime()
    current_candle = df_scaled.loc[df_scaled.index == current_time]
    
    if current_candle.empty:
        print("No candle found for the current time.")
        return None, None
    
    Close = df.loc[df.index == current_time]
    EntryPrice = Close['close'].values[0]
    
    PredictionDir = model.predict(current_candle)
    return PredictionDir, EntryPrice

import MetaTrader5 as mt5