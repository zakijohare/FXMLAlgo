import os
import joblib
import MetaTrader5 as mt5
import schedule
import time
from datetime import datetime, timedelta
from functions import ProcessAndExecute, MagTrain, DirTrain, FetchJSONNews

# Define the directory path for the Models folder on the same level
models_dir = os.path.join(os.path.dirname(__file__), 'Models')

# Define the full path for the model, scaler, PCT, NonStatMag, NonStatDir, and HighImpactNews files
mag_model_filename = os.path.join(models_dir, 'MagModel.joblib')
dir_model_filename = os.path.join(models_dir, 'DirModel.joblib')
pct_filename = os.path.join(models_dir, 'PCT.joblib')
mag_scaler_filename = os.path.join(models_dir, 'MagScaler.joblib')
dir_scaler_filename = os.path.join(models_dir, 'DirScaler.joblib')
non_stat_mag_filename = os.path.join(models_dir, 'NonStatMag.joblib')
non_stat_dir_filename = os.path.join(models_dir, 'NonStatDir.joblib')
high_impact_news_filename = os.path.join(models_dir, 'HighImpactNews.joblib')

# Load the models, scalers, PCT, NonStatMag, and NonStatDir from the files
MagModel = joblib.load(mag_model_filename)
DirModel = joblib.load(dir_model_filename)
PCT = joblib.load(pct_filename)
MagScaler = joblib.load(mag_scaler_filename)
DirScaler = joblib.load(dir_scaler_filename)
NonStatMag = joblib.load(non_stat_mag_filename)
NonStatDir = joblib.load(non_stat_dir_filename)
HighImpactNews = joblib.load(high_impact_news_filename)

def execute_job():
    current_time = datetime.now()
    if current_time.weekday() < 5 and 5 <= current_time.hour < 18:
        # Call the ProcessAndExecute function with the specified order of parameters
        ProcessAndExecute(
            symbol='XAUUSD',
            timeframe=mt5.TIMEFRAME_H1,
            MagModel=MagModel,
            DirModel=DirModel,
            DirScaler=DirScaler,
            MagScaler=MagScaler,
            NonStatMag=NonStatMag,
            NonStatDir=NonStatDir,
            PCT=PCT,
            HighImpactNews=HighImpactNews
        )

def train_models():
    # Train the models
    MagModel, PCT, MagScaler, NonStatMag = MagTrain('XAUUSD', n=24000, timeframe=mt5.TIMEFRAME_H1)
    DirModel, DirScaler, NonStatDir = DirTrain('XAUUSD', n=24000, timeframe=mt5.TIMEFRAME_H1)

    # Save the trained models and scalers
    joblib.dump(MagModel, mag_model_filename)
    joblib.dump(DirModel, dir_model_filename)
    joblib.dump(PCT, pct_filename)
    joblib.dump(MagScaler, mag_scaler_filename)
    joblib.dump(DirScaler, dir_scaler_filename)
    joblib.dump(NonStatMag, non_stat_mag_filename)
    joblib.dump(NonStatDir, non_stat_dir_filename)

    print(f"Models, scalers, and non-stationary components saved to {models_dir}")

def fetch_news_save():
    # Fetch news and save
    HighImpactNews = FetchJSONNews('https://nfs.faireconomy.media/ff_calendar_thisweek.json')
    joblib.dump(HighImpactNews, high_impact_news_filename)
    print("High-impact news data saved.")

# Schedule the job to run every hour on weekdays between 5 AM and 5 PM
schedule.every().hour.at(":00").do(execute_job)

# Schedule the model training to run every Saturday evening
schedule.every().saturday.at("17:32").do(train_models)

# Schedule the news fetching to run every Sunday morning
schedule.every().sunday.at("03:00").do(fetch_news_save)

while True:
    schedule.run_pending()
    time.sleep(1)