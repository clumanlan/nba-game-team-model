import pandas as pd
import sklearn 
import seaborn as sns
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, ScoreboardV2, BoxScoreAdvancedV2, BoxScoreTraditionalV2
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import players
from datetime import date
import awswrangler as wr
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import time
import datetime as dt
import mlflow
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error,  r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.model_selection import  cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


# TODO: validate data :) create a baseline model, setup infrastrcutre for CI/CD to SAGEMAKER WITH MLFLOW, 
##     could use pydantic to set column types of data coming in
# USE MAPE? we'll use a bunch of different metric values 

# check player dfs for missing game ids 

# FUNCTIONS ---------------------------------------

def get_game_headers() -> tuple:

    game_stats_path = "s3://nbadk-model/game_stats/game_header"

    game_headers_df = wr.s3.read_parquet(
        path=game_stats_path,
        path_suffix = ".parquet" ,
        use_threads =True
    )


    game_headers_df_processed = (game_headers_df
        .assign(
            gametype_string = game_headers_df.GAME_ID.str[:3],
            game_type = lambda x: np.where(x.gametype_string == '001', 'Pre-Season',
                np.where(x.gametype_string == '002', 'Regular Season',
                np.where(x.gametype_string == '003', 'All Star',
                np.where(x.gametype_string == '004', 'Post Season',
                np.where(x.gametype_string == '005', 'Play-In Tournament', 'unknown'))))),
            GAME_ID = game_headers_df['GAME_ID'].astype(str),
            GAME_DATE_EST = pd.to_datetime(game_headers_df['GAME_DATE_EST'])

        )
    )

    game_headers_df_processed.drop_duplicates(subset=['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID'], inplace=True)

    rel_cols = ['GAME_ID', 'game_type', 'SEASON', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS', 'HOME_TEAM_LOSSES']

    game_headers_df_processed_filtered = game_headers_df_processed[rel_cols]
    game_headers_df_processed_filtered = game_headers_df_processed_filtered.drop_duplicates()

    # Create long table in order to flag teams that are home or away
    game_home_away = game_headers_df_processed[['GAME_ID','HOME_TEAM_ID', 'VISITOR_TEAM_ID']]
    game_home_away = pd.melt(game_home_away, id_vars='GAME_ID', value_name='TEAM_ID', var_name='home_away')
    game_home_away['home_away'] = game_home_away['home_away'].apply(lambda x: 'home' if x == 'HOME_TEAM_ID' else 'away')

    return game_headers_df_processed_filtered, game_home_away


def get_player_dfs() -> tuple:
    """
    Get dataframes for player information, traditional box score stats, and advanced box score stats 
    for a given set of game IDs.

    Args:
        rel_game_ids (list): List of relevant game IDs to filter box score dataframes by.

    Returns:
        tuple: A tuple of three pandas dataframes: player_info_df, boxscore_trad_player_df, and boxscore_adv_player_df.
    """

    # Read in player information data
    player_info_path = "s3://nbadk-model/player_info"
    player_info_df = wr.s3.read_parquet(
        path=player_info_path,
        path_suffix=".parquet",
        use_threads=True
    )

    # Clean player information data and rename columns
    player_info_df = player_info_df[['PERSON_ID', 'HEIGHT', 'POSITION']].drop_duplicates()
    player_info_df = player_info_df.rename({'PERSON_ID': 'PLAYER_ID'}, axis=1)

    # Read in traditional box score data for players and filter by relevant game IDs
    boxscore_trad_player_path = "s3://nbadk-model/player_stats/boxscore_traditional/"
    boxscore_trad_player_df = wr.s3.read_parquet(
        path=boxscore_trad_player_path,
        path_suffix=".parquet",
        use_threads=True
    )
    boxscore_trad_player_df['GAME_ID'] = boxscore_trad_player_df['GAME_ID'].astype(str)

    # Read in advanced box score data for players and filter by relevant game IDs
    boxscore_adv_player_path = "s3://nbadk-model/player_stats/boxscore_advanced/"
    boxscore_adv_player_df = wr.s3.read_parquet(
        path=boxscore_adv_player_path,
        path_suffix=".parquet",
        use_threads=True
    )
    boxscore_adv_player_df = boxscore_adv_player_df.drop_duplicates(subset=['GAME_ID','PLAYER_ID'])

    return player_info_df, boxscore_trad_player_df, boxscore_adv_player_df


def get_team_level_dfs() -> tuple:
    """
    Retrieve team level dataframes for the given game IDs.

    Args:
    rel_game_ids (list): A list of game IDs to filter the dataframes by.

    Returns:
    tuple: A tuple of two pandas dataframes, the first containing traditional team stats and the second containing advanced team stats.
    """

    # Read in traditional boxscore team stats
    boxscore_trad_team_path = "s3://nbadk-model/team_stats/boxscore_traditional/"

    boxscore_trad_team_df = wr.s3.read_parquet(
        path=boxscore_trad_team_path,
        path_suffix=".parquet",
        use_threads=True
    )

    # Convert GAME_ID to string and filter by rel_game_ids
    boxscore_trad_team_df['GAME_ID'] = boxscore_trad_team_df['GAME_ID'].astype(str)
    boxscore_trad_team_df = boxscore_trad_team_df.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])

    # Read in advanced boxscore team stats
    boxscore_adv_team_path = "s3://nbadk-model/team_stats/boxscore_advanced/"

    boxscore_adv_team_df = wr.s3.read_parquet(
        path=boxscore_adv_team_path,
        path_suffix=".parquet",
        use_threads=True
    )

    # Drop duplicates and filter by rel_game_ids
    boxscore_adv_team_df = boxscore_adv_team_df.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])

    return boxscore_trad_team_df, boxscore_adv_team_df


def get_odds_data():
    odds_path = "s3://nbadk-model/oddsshark/team_odds/nba_team_odds_historical.parquet"

    odds_data = wr.s3.read_parquet(
            path=odds_path,
            path_suffix=".parquet",
            use_threads=True
        )


# check game ids,
# baseline model is just rolling average, we just store models in s3 but keep tracking local



game_headers_raw, game_home_away = get_game_headers()

game_headers_filtered = (
    game_headers_raw[game_headers_raw['game_type'].isin(['Regular Season', 'Post Season', 'Play-In Tournament'])]
    .drop(['HOME_TEAM_WINS', 'HOME_TEAM_LOSSES'], axis=1)
)

game_headers_filtered = pd.melt(game_headers_filtered, id_vars=['GAME_ID','game_type', 'SEASON', 'GAME_DATE_EST'], value_vars=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'], var_name='HOME_VISITOR', value_name='TEAM_ID')
game_headers_filtered['HOME_VISITOR'] = game_headers_filtered['HOME_VISITOR'].replace({'HOME_TEAM_ID':'HOME', 'VISITOR_TEAM_ID':'VISITOR'})


boxscore_trad_team_df, boxscore_adv_team_df = get_team_level_dfs()


trad_cols_to_remove = ['TEAM_ABBREVIATION', 'TEAM_CITY']
boxscore_trad_team_df = boxscore_trad_team_df.drop(trad_cols_to_remove, axis=1)

adv_team_cols_to_remove = ['TEAM_NAME', 'TEAM_ABBREVIATION', 'TEAM_CITY','MIN']
boxscore_adv_team_df = boxscore_adv_team_df.drop(adv_team_cols_to_remove, axis=1)

team_boxscore_combined = pd.merge(boxscore_trad_team_df, boxscore_adv_team_df, on=['GAME_ID', 'TEAM_ID'], how='left')

game_ids_missing = ['0020300778', '0022300023'] # check these later
team_boxscore_combined = team_boxscore_combined[~team_boxscore_combined['GAME_ID'].isin(game_ids_missing)] # one game id missing trad & adv stats

game_header_team_boxscore_combined = pd.merge(game_headers_filtered, team_boxscore_combined,  on=['GAME_ID', 'TEAM_ID'], how='inner')


# CREATE A REGULAR SEASON BASELINE MODEL --------------------------------------------------------------------------

game_team_regular = (
    game_header_team_boxscore_combined[game_header_team_boxscore_combined['game_type']=='Regular Season']
    .assign(
        TEAM_ID = lambda x: x['TEAM_ID'].astype(str)
    )
    .sort_values(['GAME_DATE_EST', 'TEAM_ID'])
    .reset_index(drop=True)
)

game_team_regular_train = game_team_regular[game_team_regular['SEASON']<2019].copy()

for i in range(1,6):

    ## accuracy flatlines at about 5 games out, with each game out happening there after only reducing by 0.1
    col_label = f'team_lagged_pts_rolling_{i}_mean'
    game_team_regular_train.loc[:,col_label] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])['PTS'].transform(lambda x: x.shift(1).rolling(i, min_periods=i).mean())

    print(col_label)



game_team_regular_five = game_team_regular_train[~game_team_regular_train['team_lagged_pts_rolling_5_mean'].isnull()]
rolling_avg_five = game_team_regular_five['team_lagged_pts_rolling_5_mean']
actual = game_team_regular_five['PTS']

baseline_mse = mean_squared_error(actual, rolling_avg_five)
baseline_rmse = np.sqrt(mean_squared_error(actual, rolling_avg_five))
baseline_mae = mean_absolute_error(actual, rolling_avg_five)



# PROCESS FEATURES -----------------------------------------------------------


# create a simple model first with year, rolling averages, and home and away

# numeric feature processing --------------------------------------------------
rel_num_feats = ['SEASON'] + [col for col in game_team_regular_train.columns if 'pts_rolling' in col]

num_pipeline = Pipeline(steps=[
    ('scale', StandardScaler())
])



# cat feature processing ------------------------------------------------------------
cat_cols_high_card = ['PLAYER_ID', 'TEAM_ID']

cat_pipeline_high_card = Pipeline(steps=[
    ('encoder', TargetEncoder(smoothing=2))
])


rel_cat_feats_low_card = ['HOME_VISITOR']

cat_pipeline_low_card = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])




# date feature processing ---------------------------------------------------------
date_feats = ['dayofweek', 'dayofyear',  'is_leap_year', 'quarter', 'weekofyear', 'year']

class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
          return self

    def transform(self, x):

        x['GAME_DATE_EST'] = pd.to_datetime(x['GAME_DATE_EST'])

        dayofweek = x.GAME_DATE_EST.dt.dayofweek
        dayofyear= x.GAME_DATE_EST.dt.dayofyear
        is_leap_year =  x.GAME_DATE_EST.dt.is_leap_year
        quarter =  x.GAME_DATE_EST.dt.quarter
        weekofyear = x.GAME_DATE_EST.dt.weekofyear
        year = x.GAME_DATE_EST.dt.year

        df_dt = pd.concat([dayofweek, dayofyear,  is_leap_year, quarter, weekofyear, year], axis=1)

        return df_dt

date_pipeline = Pipeline(steps=[
    ('date', DateTransformer())
])



col_trans_pipeline = ColumnTransformer(
    transformers=[
        ('date', date_pipeline, ['GAME_DATE_EST']),
        ('numeric', num_pipeline, rel_num_feats),
        ('cat_low', cat_pipeline_low_card, rel_cat_feats_low_card)
    ]
)

model = LinearRegression()

pipeline = Pipeline(steps=[
    ('preprocess', col_trans_pipeline),
    ('model', model)
])



# Mlflow Tracking  ---------------------------------------------------


# mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db  --default-artifact-root mlruns/  --artifacts-destination s3://nbadk-model/models/team-game/experiments

local_server_uri = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(local_server_uri)

experiment_name = "team_game"

#mlflow.create_experiment(experiment_name, artifact_location="s3://nbadk-model/models/team-game/experiments")
mlflow.set_experiment(experiment_name)

with mlflow.start_run():

    model_name_tag = "baseline_team_rolling_5_game_avg"
    
    mlflow.set_tag('mlflow.runName', model_name_tag) 
    mlflow.set_tag("mlflow.note.content", "We determined that 5 games is where RMSE error levels out, so we filter out first 4 outcome obs going forward")
    

    train_filtered = game_team_regular_train.dropna(subset=['team_lagged_pts_rolling_5_mean']).reset_index(drop=True)

    X_train = train_filtered[rel_num_feats + rel_cat_feats_low_card + ['GAME_DATE_EST']]
    y_train = train_filtered['PTS']

    mlflow.log_param("model_name", model_name_tag)
    #mlflow.log_param("features_used", ", ".join(X_train.columns))
    mlflow.log_param("features_used", "none")
    


    #cross_val_scores = cross_val_score(pipeline, X_train, y_train, cv=tscv)
    #cross_val_score_mean_sqaured_error = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')

    #pipeline.fit(X_train, y_train)
    #y_train_pred = pipeline.predict(X_train) 
    #adjusted_r2 = 1 - ( 1-r2 ) * (len(y_train) - 1 ) / ( len(y_train) - X_train.shape[1] - 1 )
    
    mlflow.log_metric("mse", baseline_mse)
    mlflow.log_metric("rmse", baseline_rmse)
    mlflow.log_metric("mae", baseline_mae)

    mlflow.log_metric('cv_neg_mse', baseline_mse)
    mlflow.log_metric('cv_rmse', baseline_rmse)


    #mlflow.log_metric('adjusted_r2', adjusted_r2)
    #mlflow.log_metric('cv_neg_mse', cross_val_scores.mean())
    #mlflow.log_metric('cv_rmse', np.mean(np.sqrt(np.abs(cross_val_score_mean_sqaured_error))))

    #mlflow.sklearn.log_model(pipeline, model_name_tag)








   
