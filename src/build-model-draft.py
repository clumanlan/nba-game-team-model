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
from sklearn.metrics import mean_squared_error, mean_absolute_error,  r2_score, make_scorer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

pd.set_option('display.max_columns', None)

# TODO: validate data :) create a baseline model, setup infrastrcutre for CI/CD to SAGEMAKER WITH MLFLOW, 
##     could use pydantic to set column types of data coming in
# USE MAPE? we'll use a bunch of different metric values 


# just basically the way this flow works is you create a feature you trend it out, the question is two fold right:
# # what features matter and over what period of time


# much later date can bring in PLAYER DFS
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

def get_sec(time_str):
    """Get seconds from time."""
    if ':' in time_str:
        if '.' in time_str:
            time_str = time_str.replace('.000000', '')
            m, s = time_str.split(':')
            time_sec = int(m) *60 + int(s)
        else:
            m, s = time_str.split(':')
            time_sec = int(m)*60 + int(s)
    
    if ':' not in time_str:
        if time_str == 'None':
            time_sec = 0
        else: 
            time_sec = int(time_str)*60

    return time_sec


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


# CREATE BASEIC FEATS AND SPLIT INTO A TRAINING DATAFRAME --------------------------------------------------------------------------

game_team_regular = (
    game_header_team_boxscore_combined[game_header_team_boxscore_combined['game_type']=='Regular Season']
    .assign(
        TEAM_ID = lambda x: x['TEAM_ID'].astype(str),
        SEC = lambda x: x['MIN'].apply(get_sec)
    )
    .sort_values(['GAME_DATE_EST', 'TEAM_ID'])
    .drop(['MIN'], axis=1)
    .reset_index(drop=True)
)



game_team_regular_train = game_team_regular[game_team_regular['SEASON']<2019].copy()


# ADD OPPOSING TEAM STATS (DEFENSIVE PERFORMANCE) -------------------------------------------------------------------------

# to get allowed stats we create two data frames GAME_ID, TEAM_ID for HOME AND AWAY, then take home and join to away stats by GAME_ID 
gr_train_home_base = game_team_regular_train[game_team_regular_train['HOME_VISITOR']=='HOME'][['TEAM_ID', 'GAME_ID']]
gr_train_away_base = game_team_regular_train[game_team_regular_train['HOME_VISITOR']=='VISITOR'][['TEAM_ID', 'GAME_ID']]


non_rel_opposing_cols = ['game_type', 'SEASON', 'GAME_DATE_EST', 'HOME_VISITOR', 'TEAM_ID', 'TEAM_NAME', 'OREB_PCT',
                         'DREB_PCT', 'REB_PCT']

gr_train_home_stats = game_team_regular_train[game_team_regular_train['HOME_VISITOR']=='HOME'].drop(non_rel_opposing_cols, axis=1)
gr_train_away_stats = game_team_regular_train[game_team_regular_train['HOME_VISITOR']=='VISITOR'].drop(non_rel_opposing_cols, axis=1)


gr_train_home = pd.merge(gr_train_home_base, gr_train_away_stats, on='GAME_ID')
gr_train_home.columns = [f"{col}_allowed_opposing" for col in gr_train_home.columns]
gr_train_home = gr_train_home.rename({'TEAM_ID_allowed_opposing': 'TEAM_ID', 'GAME_ID_allowed_opposing': 'GAME_ID'}, axis=1)


gr_train_away = pd.merge(gr_train_away_base, gr_train_home_stats, on='GAME_ID')
gr_train_away.columns = [f"{col}_allowed_opposing" for col in gr_train_away.columns]
gr_train_away = gr_train_away.rename({'TEAM_ID_allowed_opposing': 'TEAM_ID', 'GAME_ID_allowed_opposing': 'GAME_ID'}, axis=1)


gr_train_opposing = pd.concat([gr_train_away, gr_train_home])


game_team_regular_train = pd.merge(game_team_regular_train, gr_train_opposing, how='left', on=['TEAM_ID', 'GAME_ID'])

# the problem is i don't understand some of these stats and how they affect things: make a basic decision tree!
# so i should see what features are importnat and if removing some of these that i don't understand affects performance
# 


# ranking points allowed and points made: ranking of across the season makes sense and honestly last 10 games,




## see if perofrmance improves and either way i need to figure out how to whiteboard and think of features that will actually have an impact
## honestly i'd be pretty at peace with 3-5 rmse off 


# TIME TREND: WE CREATE TWO TYPES: SHORT AND LONG FOR ANY FEATURES WE CREATE ---------------------------------------------
static_cols = ['game_type', 'SEASON', 'GAME_DATE_EST', 'HOME_VISITOR']
lagged_num_cols = ['PTS', 'SEC', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
       'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
       'BLK', 'TO', 'PF',  'PLUS_MINUS', 'E_OFF_RATING', 'OFF_RATING',
       'E_DEF_RATING', 'DEF_RATING', 'E_NET_RATING', 'NET_RATING', 'AST_PCT',
       'AST_TOV', 'AST_RATIO', 'E_TM_TOV_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'USG_PCT',
       'E_USG_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE']

lagged_num_cols_opposing = [f"{col}_allowed_opposing" for col in lagged_num_cols]

contains_sig_nulls = ['OREB_PCT', 'DREB_PCT', 'REB_PCT']

lagged_num_cols_complete = lagged_num_cols + lagged_num_cols_opposing



# SHORT TREND ------------------------------------------------------------------------------------------

## num feats trend -----------
loop_place=0

for col in lagged_num_cols_complete:
    
    temp_lagged_col_df = pd.DataFrame()

    for i in range(2,6):

        ## accuracy flatlines at about 5 games out, with each game out happening there after only reducing by 0.1
        lagged_col_label = f'team_lagged_{col}'
        temp_lagged_col_df[lagged_col_label] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])[col].transform(lambda x: x.shift(1))

        mean_col_label = f'team_lagged_{col}_rolling_{i}_mean'
        temp_lagged_col_df[mean_col_label] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])[col].transform(lambda x: x.shift(1).rolling(i, min_periods=i).mean())

        median_col_label = f'team_lagged_{col}_rolling_{i}_median'
        temp_lagged_col_df[median_col_label] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])[col].transform(lambda x: x.shift(1).rolling(i, min_periods=i).median())

        std_col_label = f'team_lagged_{col}_rolling_{i}_std'
        temp_lagged_col_df[std_col_label]  = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])[col].transform(lambda x: x.shift(1).rolling(i, min_periods=i).std())
    
    game_team_regular_train = pd.concat([game_team_regular_train, temp_lagged_col_df], axis=1)

    loop_place+=1

    pct_complate = loop_place/len(lagged_num_cols_complete)
    print("{:.2%}".format(pct_complate))

    del temp_lagged_col_df



## cat feats trend --------------
lagged_cat_cols = ['HOME_VISITOR']

game_team_regular_train = game_team_regular_train.set_index('GAME_DATE_EST')
game_team_regular_train['home'] = np.where(game_team_regular_train['HOME_VISITOR']=='HOME', 1, 0)
game_team_regular_train['away'] = np.where(game_team_regular_train['HOME_VISITOR']=='VISITOR', 1, 0)


window_size = ['7D', '14D', '30D']

loop_place = 0
for window in window_size:

    temp_lagged_cat_col_df = pd.DataFrame()

    temp_lagged_cat_col_df[f'game_count_{window}'] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])['GAME_ID'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).count())


    temp_lagged_cat_col_df[f'home_game_count_{window}'] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])['home'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())


    temp_lagged_cat_col_df[f'away_game_count_{window}'] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])['away'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())

    temp_lagged_cat_col_df = temp_lagged_cat_col_df.fillna(0)

    game_team_regular_train = pd.concat([game_team_regular_train,temp_lagged_cat_col_df], axis=1)

    del temp_lagged_cat_col_df

    loop_place+=1
    pct_complate = loop_place/len(window_size)
    print("{:.2%}".format(pct_complate))


game_team_regular_train = game_team_regular_train.reset_index()



# LONG TREND - WHOLE SEASON ROLLING -------------------------------------------------------------------------------

## num feats trend ---------------------------
loop_place=0

for col in lagged_num_cols_complete:
    
    temp_lagged_col_df = pd.DataFrame()

    mean_col_label = f'team_lagged_{col}_rolling_season_mean'
    temp_lagged_col_df[mean_col_label] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])[col].transform(lambda x: x.shift(1).rolling(100, min_periods=i).mean())

    median_col_label = f'team_lagged_{col}_rolling_season_median'
    temp_lagged_col_df[median_col_label] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])[col].transform(lambda x: x.shift(1).rolling(100, min_periods=i).median())

    std_col_label = f'team_lagged_{col}_rolling_season_std'
    temp_lagged_col_df[std_col_label]  = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])[col].transform(lambda x: x.shift(1).rolling(100, min_periods=i).std())
    
    game_team_regular_train = pd.concat([game_team_regular_train, temp_lagged_col_df], axis=1)

    loop_place+=1

    pct_complate = loop_place/len(lagged_num_cols_complete)
    print("{:.2%}".format(pct_complate))

    del temp_lagged_col_df




## cat feats trend ---------------------------
temp_lagged_cat_col_df = pd.DataFrame()

temp_lagged_cat_col_df['days_since_last_game'] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])['GAME_DATE_EST'].diff().dt.days

temp_lagged_cat_col_df[f'game_count_season'] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])['GAME_ID'].transform(lambda x: x.shift(1).rolling(window=100, min_periods=1).count())


temp_lagged_cat_col_df[f'home_game_count_season'] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])['home'].transform(lambda x: x.shift(1).rolling(window=100, min_periods=1).sum())


temp_lagged_cat_col_df[f'away_game_count_season'] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])['away'].transform(lambda x: x.shift(1).rolling(window=100, min_periods=1).sum())

temp_lagged_cat_col_df = temp_lagged_cat_col_df.fillna(0)

game_team_regular_train = pd.concat([game_team_regular_train,temp_lagged_cat_col_df], axis=1)

del temp_lagged_cat_col_df


# CREATE ROLLING RANKINGS  -----------------------------------------------------------------

# USE LAGGED FEATS TO CREATE RANKINGS: OFFENSE AND DEFENSE (Which is just opposing)
## you create rankings of differnt things which is rather interesting 




lagged_num_cols_complete.remove('PTS')


game_team_regular_train_filtered = (
    game_team_regular_train
    .drop(lagged_num_cols_complete + contains_sig_nulls + ['home','away'], axis=1)
    .dropna(subset=['team_lagged_PTS_rolling_5_mean'])
    .reset_index(drop=True)
)




game_team_regular_train[(game_team_regular_train['TEAM_ID']=='1610612739') & (game_team_regular_train['SEASON']==2018)][['GAME_DATE_EST', 'days_since_last_game','game_count_7D','home_game_count_7D', 'away_game_count_7D']].head(10)



# BASELINE MODEL -------------------------------------------------------------

## accuracy flatlines at about 5 games out, with each game out happening there after only reducing by 0.1
## that being said this is for the baseline model with just rolling averages of target variable and no other features + interactions 

rolling_avg_five = game_team_regular_train_filtered['team_lagged_PTS_rolling_5_mean']
actual = game_team_regular_train_filtered['PTS']

baseline_mse = mean_squared_error(actual, rolling_avg_five)
baseline_rmse = np.sqrt(mean_squared_error(actual, rolling_avg_five))
baseline_mae = mean_absolute_error(actual, rolling_avg_five)



# PROCESS FEATURES -----------------------------------------------------------

# create a simple model first with year, rolling averages, and home and away

# numeric feature processing --------------------------------------------------
rel_num_feats = game_team_regular_train_filtered.select_dtypes(include=np.number).columns.tolist()
rel_num_feats.remove('PTS')

num_pipeline = Pipeline(steps=[
    ('scale', StandardScaler())
])



# cat feature processing ------------------------------------------------------------
cat_cols_high_card = ['PLAYER_ID', 'TEAM_ID']

cat_pipeline_high_card = Pipeline(steps=[
])

#    ('encoder', TargetEncoder(smoothing=2))

rel_cat_feats_low_card = ['HOME_VISITOR']

cat_pipeline_low_card = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])




# date feature processing ---------------------------------------------------------
date_feats = ['dayofweek', 'dayofyear',  'is_leap_year', 'quarter',  'year']

class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
          return self

    def transform(self, x):

        x['GAME_DATE_EST'] = pd.to_datetime(x['GAME_DATE_EST'])

        dayofweek = x.GAME_DATE_EST.dt.dayofweek
        dayofyear= x.GAME_DATE_EST.dt.dayofyear
        is_leap_year =  x.GAME_DATE_EST.dt.is_leap_year
        quarter =  x.GAME_DATE_EST.dt.quarter
        #weekofyear = x.GAME_DATE_EST.dt.weekofyear
        year = x.GAME_DATE_EST.dt.year

        df_dt = pd.concat([dayofweek, dayofyear,  is_leap_year, quarter,  year], axis=1)

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

# CREATE TRAIN DF WE'LL USE RANDOM SAMPLE FROM TRAIN DF -----------------------------------------------

train_sample = game_team_regular_train_filtered.sample(20000).reset_index(drop=True)

X_train = train_sample[rel_num_feats + rel_cat_feats_low_card + ['GAME_DATE_EST']]
y_train =  train_sample['PTS']


# TRY ON A SINGLE TREE WITH VARYING DEPTHS TO SEE WHAT IT LOOKS LIKE ----------------------------
from sklearn import tree

single_tree = RandomForestRegressor(n_estimators=1, max_depth=6,
                          bootstrap=False, n_jobs=-1)

single_tree_pipeline = Pipeline(steps=[
    ('preprocess', col_trans_pipeline),
    ('model', single_tree)
])

single_tree_pipeline.fit(X_train, y_train)

y_train_pred = single_tree_pipeline.predict(X_train)
r2_score(y_train, y_train_pred)


num_feats = single_tree_pipeline.named_steps['preprocess'].transformers_[1][2]
cat_feats = single_tree_pipeline.named_steps['preprocess'].transformers_[2][1].named_steps['encoder'].get_feature_names_out().tolist()

feat_names = date_feats + num_feats + cat_feats

tree.plot_tree(single_tree_pipeline['model'][0],
                feature_names=feat_names,
                filled=True,
                rounded=True,
                fontsize=6)


from dtreeviz.trees import dtreeviz

viz = dtreeviz(single_tree_pipeline['model'][0], X_train, y_train, feature_names=feat_names, target_name="PTS")
viz


## random forest introduces randomness to it and averages a bunch of uncorrelated trees,
## randomness is introduced by bootstrap sampling rows and selecting only a few columns to choose each split by


# Mlflow Tracking  ---------------------------------------------------


# mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db  --default-artifact-root mlruns/  --artifacts-destination s3://nbadk-model/models/team-game/experiments

local_server_uri = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(local_server_uri)

experiment_name = "team_game"

#mlflow.create_experiment(experiment_name, artifact_location="s3://nbadk-model/models/team-game/experiments")
mlflow.set_experiment(experiment_name)

with mlflow.start_run():


    #mlflow.sklearn.autolog() # need to check out what this records
    model_name_tag = "rf_v3_log_pts"
    
    #model = LinearRegression()
    model = RandomForestRegressor(random_state=32, n_estimators=10, min_samples_leaf=100, n_jobs=-1) # default n_estimators is 100
    #model = xgb.XGBRegressor(random_state=32)

    pipeline = Pipeline(steps=[
        ('preprocess', col_trans_pipeline),
        ('model', model)
    ])

    
    mlflow.set_tag('mlflow.runName', model_name_tag) 
    mlflow.set_tag("mlflow.note.content", "V3 model add opposing teams stats and short + long term window sizes")
    
    y_train_log = np.log(y_train)
    pipeline.fit(X_train, y_train_log)
    y_train_pred = pipeline.predict(X_train) 
    
    mse = mean_squared_error(y_train, y_train_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, y_train_pred)


    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)

    print(f"MSE {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")


    mlflow.sklearn.log_model(pipeline, model_name_tag)



r2_score(y_train, y_train_pred)

# EXPLORE DISTRIBUTION OF ERRORS AFTER TRAINING ------------------------------------------

train_pred_df = pd.DataFrame({'observed': y_train, 'predicted': y_train_pred})
train_pred_df['residual'] = train_pred_df['predicted']-train_pred_df['observed']

# look at distributionof y_train
px.histogram(train_pred_df, x='observed')

# look at distribution of residual
px.histogram(train_pred_df, x='residual')


# observe versus predicted 
px.scatter(data_frame=train_pred_df, x='predicted', y='observed')



# residual versus predicted 
px.scatter(data_frame=train_pred_df, x='predicted', y='residual')




# EXPLORE FEATURE IMPORTANCE AFTER TRAINING ---------------------------------------

# random forest feature importance

from rfpimp import *  # feature importance plot

rf_importances = importances(pipeline, X_train, y_train)
plot_importances(rf_importances.head(30))




# I THINK THIS IS DONE LATER ------------------------------------
# training error versus number of train records?

# plot training error versus parameter complexity? --- this would be number of trees, or min_leaf_size for random forest and xgboost



# RUN MODEL OVER SIMULATION OF SPORTSBOOK OUTCOME -----------

from sklearn.inspection import permutation_importance

# we'll save cross validation for the final steps here ------------------------

tscv = TimeSeriesSplit(n_splits=5)

cross_val_scores_r2 = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring=make_scorer(r2_score))
cross_val_score_mse = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
cross_val_score_mae = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')

# mlflow.log_metric('cv_r2', cross_val_scores_r2.mean())
# mlflow.log_metric('cv_neg_mse', cross_val_score_mse.mean())
# mlflow.log_metric('cv_rmse', np.mean(np.sqrt(np.abs(cross_val_score_mse))))
# mlflow.log_metric('cv_mae', cross_val_score_mae.mean())
