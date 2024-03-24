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
import mlflow
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error,  r2_score, make_scorer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from datetime import timedelta
pd.set_option('display.max_columns', None)

# TODO: validate data :) create a baseline model, s
# add features: distance from home
# Setup infrastrcutre for CI/CD to SAGEMAKER WITH MLFLOW, 
##     could use pydantic to set column types of data coming in
# USE MAPE? we'll use a bunch of different metric values 


# just basically the way this flow works is you create a feature you trend it out, the question is two fold right:
# # what features matter and over what period of time


# much later date can bring in PLAYER DFS
# check player dfs for missing game ids 

from buildmodelmodule.classes import GetData, TransformData, CreateTimeBasedFeatures


# CREATE INITIAL FEATURES ----------------------------------------------


# initialize the object:
get_data = GetData()
transform_data = TransformData()

# GET DATA FUNCTIONS -------------------------------------------------
game_headers = get_data.get_game_headers_historical()
player_info_df, boxscore_trad_player_df, boxscore_adv_player_df = get_data.get_player_dfs()
boxscore_trad_team_df, boxscore_adv_team_df = get_data.get_team_level_dfs()


# TRANSFORM DATA FUNCTIONS ------------------------------------------
game_team_regular_season = transform_data.create_reg_season_game_boxscore(game_headers, boxscore_trad_team_df, boxscore_adv_team_df)

game_team_regular_season = transform_data.process_regular_team_boxscore(game_team_regular_season)





game_team_regular_season


game_team_regular_train = game_team_regular_season[game_team_regular_season['SEASON']<2019].copy()


game_team_regular_train.GAME_DATE_WEEK_START



## cat feats trend ---------------------------------
cat_cols = ['HOME_VISITOR']

game_team_regular_train = game_team_regular_train.set_index('GAME_DATE_EST')


window_size = ['7D', '14D', '30D']

loop_place = 0
    
temp_lagged_cat_col_df = pd.DataFrame()

for window in window_size:

    temp_lagged_cat_col_df[f'GAME_count_{window}'] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])['GAME_ID'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).count())


    temp_lagged_cat_col_df[f'HOME_GAME_count_{window}'] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])['HOME'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())


    temp_lagged_cat_col_df[f'AWAY_GAME_count_{window}'] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])['AWAY'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())

    temp_lagged_cat_col_df = temp_lagged_cat_col_df.fillna(0)

    loop_place+=1
    pct_complate = loop_place/len(window_size)
    print("{:.2%}".format(pct_complate))

game_team_regular_train = pd.concat([game_team_regular_train,temp_lagged_cat_col_df], axis=1).reset_index()

del temp_lagged_cat_col_df



game_team_regular_train_filtered = game_team_regular_season[static_cols + lagged_num_cols_complete + cat_cols]




# READ IN STADIUM DISTANCE DATAFRAME

nba_stadiums_path = "s3://nbadk-model/stadiuminfo/stadium_distances/"

nba_stadiums_df = wr.s3.read_parquet(
    path=nba_stadiums_path,
    path_suffix = ".parquet" ,
    use_threads =True
)

# enter max, min of all of num columns as well over different time frames,
# FIGURE OUT HOW TO INSERT EXPONENTIAL SMOOTHING AND AUTOARIMA INTO NUM FEATS only for a few right: PTS, PTS_ALLOWED, POSSESIONS

 'FGM', 'FGA', 'FG_PCT', 

'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 

odds_data = get_data.get_odds_data()
kaggle_odds_data = get_data.get_odds_data_kaggle()



game_stats_rolling_path = "s3://nbadk-model/game_stats/game_header/rolling"

game_headers_df_rolling = wr.s3.read_parquet(
    path=game_stats_rolling_path,
    path_suffix = ".parquet" ,
    use_threads =True
)



# PROCESS HISTORICAL DATA ----------------------------------------
# CREATE A FUNCITON LATER THAT PROCESSED ROLLING HISTORICAL WHICH WOULD JUST BE 2024>= then appended on


# the problem is i don't understand some of these stats and how they affect things: make a basic decision tree!
# 3-5 RMSE IS GOAL 
# ranking points allowed and points made: ranking of across the season makes sense and honestly last 10 games,



# FINAL TRANSFORM -------------------------------------------------------------------------------



# CREATE FEATS ------------------------------------------------------------------------------------------



# EXPONENTIAL SMOOTH SKTIME ARIMA OF GIVEN COLUMNS
from sktime.registry import all_estimators

all_estimators("forecaster", as_dataframe=True)

rel_sktime_transformer = all_estimators(estimator_types="transformer",filter_tags='capability:unequal_length', as_dataframe=True)
rel_sktime_transformer

# ROLLING WEEK STATS THEN YOU FORECAST IT OUT 
game_team_regular_train['TEAM_ID'] = game_team_regular_train['TEAM_ID'].astype(str)
game_team_regular_train['GAME_DATE_EST'] = pd.to_datetime(game_team_regular_train['GAME_DATE_EST'])



season_week_calendar = game_team_regular_train[['SEASON','GAME_DATE_WEEK_START']].drop_duplicates()

team_weekly_pts = game_team_regular_train.groupby(['GAME_DATE_WEEK_START', 'TEAM_ID'])['PTS'].mean()

game_team_regular_train_filtered = game_team_regular_train_filtered.set_index(['TEAM_ID_SEASON', 'GAME_DATE_EST']).sort_index()



team_pts_regular_index = game_team_regular_train[game_team_regular_train['SEASON']==2018][['GAME_DATE_EST', 'TEAM_ID','PTS']].set_index(['TEAM_ID','GAME_DATE_EST'])
team_pts_regular_index = team_pts_regular_index.sort_index()


# week stats lagged!!!!

# CREATE BASEIC FEATS AND SPLIT INTO A TRAINING DATAFRAME --------------------------------------------------------------------------


# TIME TREND: WE CREATE TWO TYPES: SHORT AND LONG FOR ANY FEATURES WE CREATE ---------------------------------------------




# LONG TREND - WHOLE SEASON ROLLING -------------------------------------------------------------------------------

## num feats trend ---------------------------
loop_place=0


for col in lagged_num_cols_complete:

    temp_lagged_col_df = pd.DataFrame()

    for stat_type in ['mean', 'median', 'std']:
        
        col_label = f'team_lagged_{col}_rolling_season_{stat_type}'
        temp_lagged_col_df[col_label] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])[col].transform(lambda x: x.shift(1).rolling(100, min_periods=1).agg(stat_type))
    

    game_team_regular_train = pd.concat([game_team_regular_train, temp_lagged_col_df], axis=1)

    loop_place+=1

    pct_complate = loop_place/len(lagged_num_cols_complete)
    print("{:.2%}".format(pct_complate))


del temp_lagged_col_df


## num feats trend - home_visitor --------------------------

loop_place=0

home_visitor_stats_track = ['PTS', 'PTS_allowed_opposing']

temp_lagged_col_df = pd.DataFrame()

for col in home_visitor_stats_track:

    for stat_type in ['mean', 'median']:

        col_label = f'team_lagged_home_visitor_{col}_rolling_season_{stat_type}'
        temp_lagged_col_df[col_label] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON', 'HOME_VISITOR'])[col].transform(lambda x: x.shift(1).rolling(100, min_periods=1).agg(stat_type))

    loop_place+=1

    pct_complate = loop_place/len(home_visitor_stats_track)
    print("{:.2%}".format(pct_complate))
    

game_team_regular_train = pd.concat([game_team_regular_train, temp_lagged_col_df], axis=1)


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










# SHORT TREND ------------------------------------------------------------------------------------------

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import ForecastingPipeline
from sktime.transformations.series.summarize import WindowSummarizer


kwargs = {
    "lag_feature": {
        "lag": [1,2,3,4,5,6],
        "mean": [[1,2], [1, 3], [1, 4], [1, 5], [1, 6]],
        "median": [[1,2], [1, 3], [1, 4], [1, 5], [1, 6]],
        "std": [[1, 2],[1, 3],[1, 4],[1, 5],[1, 6]],
    }
}



transformer = WindowSummarizer(**kwargs, target_cols=lagged_num_cols)
game_team_regular_train_transformed = transformer.fit_transform(game_team_regular_train_filtered)

game_team_regular_train_transformed.head(10)













# CREATE ROLLING RANKINGS  -----------------------------------------------------------------
## we basically creates a calendar (date time index) recording that team's stats for that given point

base_ranking_cols = ['TEAM_ID', 'SEASON', 'GAME_DATE_EST']
rel_ranking_cols = ['team_lagged_PTS_rolling_season_mean', 'team_lagged_PTS_allowed_opposing_rolling_season_mean']

game_team_regular_train.columns.tolist()
team_season_calendar_list = []

team_season_ranking_base = game_team_regular_train[base_ranking_cols + rel_ranking_cols]

def reindex_by_date(df):
    dates = pd.date_range(df.GAME_DATE_EST.min(), df.GAME_DATE_EST.max())
    return df.reindex(dates).ffill()

for season in game_team_regular_train.SEASON.unique():

    df = team_season_ranking_base[team_season_ranking_base['SEASON']==season]
    df.index = pd.DatetimeIndex(df.GAME_DATE_EST)

    df = df.groupby('TEAM_ID').apply(reindex_by_date).reset_index(0, drop=True)
    team_season_calendar_list.append(df)

    print(season)

team_season_ranking_calendar_df = pd.concat(team_season_calendar_list).reset_index(names='calendar_date')

team_season_ranking_calendar_df = (team_season_ranking_calendar_df
                                   .dropna(subset=rel_ranking_cols)
                                   .drop('GAME_DATE_EST', axis=1)
)



for stat in rel_ranking_cols:
    team_season_ranking_calendar_df[f'{stat}_rank_overall'] = team_season_ranking_calendar_df.groupby('calendar_date')[stat].rank(ascending=False)


team_season_ranking_calendar_df = team_season_ranking_calendar_df.drop(['SEASON'] + rel_ranking_cols, axis=1)

game_team_regular_train = pd.merge(game_team_regular_train, team_season_ranking_calendar_df, left_on= ['GAME_DATE_EST', 'TEAM_ID'], right_on =['calendar_date', 'TEAM_ID'], how='left')


# DOUBLE CHECK THIS !!!!
game_team_regular_train[['team_lagged_PTS_rolling_season_mean_rank_overall', 'team_lagged_PTS_allowed_opposing_rolling_season_mean_rank_overall']]

# team_season_ranking_calendar_df[team_season_ranking_calendar_df['calendar_date']=='1999-11-07'].sort_values('team_lagged_PTS_rolling_season_mean')

#MORE TO DO:
## you could do this grouped by home and visitor as well, then also do this for other stats
# when did some teams change stadiums? -- so in theory we'd actually have year as a column and merge on team names nad columns


del team_season_ranking_base, team_season_calendar_list, team_season_ranking_calendar_df





game_team_visitor_base = game_team_regular_train[['GAME_ID', 'TEAM_NAME', 'HOME_VISITOR']]

game_team_visitor_base = game_team_visitor_base.merge(game_team_visitor_base, on='GAME_ID')
game_team_visitor_base = game_team_visitor_base[(game_team_visitor_base['TEAM_NAME_x'] != game_team_visitor_base['TEAM_NAME_y']) & (game_team_visitor_base['HOME_VISITOR_x']=='VISITOR')]


game_team_visitor_base = pd.merge(game_team_visitor_base, nba_stadiums_df, left_on=['TEAM_NAME_x','TEAM_NAME_y'], right_on=['TEAM_NAME_a', 'TEAM_NAME_b'])

game_team_visitor_base = game_team_visitor_base[['GAME_ID', 'distance_miles']]
game_team_visitor_base['HOME_VISITOR'] = 'VISITOR'

game_team_regular_train = game_team_regular_train.merge(game_team_visitor_base, on=['GAME_ID', 'HOME_VISITOR'], how='left')
game_team_regular_train['distance_miles'] = game_team_regular_train['distance_miles'].fillna(0)




# WE DROP FEATS HERE --------------------------------

lagged_num_cols_complete.remove('PTS')

game_team_regular_train_filtered = (
    game_team_regular_train
    .drop(lagged_num_cols_complete + contains_sig_nulls + ['home','away'], axis=1)
    .dropna(subset=['team_lagged_PTS_rolling_5_mean', 'team_lagged_home_visitor_PTS_rolling_season_mean']) 
    .reset_index(drop=True)
)

cols_with_nulls = game_team_regular_train_filtered.columns[game_team_regular_train_filtered.isnull().any()]

for column in cols_with_nulls:
    null_count = game_team_regular_train_filtered[column].isnull().sum()
    print(f"Column '{column}' has {null_count} null values.")

## hawks and 76ers spend first five games of season away


# BASELINE MODEL -------------------------------------------------------------

## accuracy flatlines at about 5 games out, with each game out happening there after only reducing by 0.1

rolling_avg_five = game_team_regular_train_filtered['team_lagged_PTS_rolling_5_mean']
actual = game_team_regular_train_filtered['PTS']

baseline_mse = mean_squared_error(actual, rolling_avg_five)
baseline_rmse = np.sqrt(mean_squared_error(actual, rolling_avg_five))
baseline_mae = mean_absolute_error(actual, rolling_avg_five)


# PROCESS FEATURES -----------------------------------------------------------

# numeric feature processing --------------------------------------------------
rel_num_feats = game_team_regular_train_filtered.select_dtypes(include=np.number).columns.tolist()
rel_num_feats.remove('PTS')

num_pipeline = Pipeline(steps=[
    ('scale', StandardScaler())
])
rel_num_feats


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

# we're going to switch this and do a train and valid set


train = game_team_regular_train_filtered[game_team_regular_train_filtered['SEASON']<2016]
valid = game_team_regular_train_filtered[game_team_regular_train_filtered['SEASON']>=2016]

X_train = train[rel_num_feats + rel_cat_feats_low_card + ['GAME_DATE_EST']]
y_train =  train['PTS']

X_valid = valid[rel_num_feats + rel_cat_feats_low_card + ['GAME_DATE_EST']]
y_valid =  valid['PTS']


# Mlflow Tracking  ---------------------------------------------------


# mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db  --default-artifact-root mlruns/  --artifacts-destination s3://nbadk-model/models/team-game/experiments

local_server_uri = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(local_server_uri)

experiment_name = "team_game"

#mlflow.create_experiment(experiment_name, artifact_location="s3://nbadk-model/models/team-game/experiments")
mlflow.set_experiment(experiment_name)

with mlflow.start_run():


    #mlflow.sklearn.autolog() # need to check out what this records
    model_name_tag = "xgb_v4_basic_pts_rankings_n_stadium_distance"
    
    #model = LinearRegression()
    #model = RandomForestRegressor(random_state=32, n_estimators=10, min_samples_leaf=100, n_jobs=-1) # default n_estimators is 100
    model = xgb.XGBRegressor(random_state=32)

    pipeline = Pipeline(steps=[
        ('preprocess', col_trans_pipeline),
        ('model', model)
    ])

    
    mlflow.set_tag('mlflow.runName', model_name_tag) 
    mlflow.set_tag("mlflow.note.content", "V4 Basic Rankings and distnace metrics")
    
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train) 
    y_valid_pred = pipeline.predict(X_valid)
    
    mse = mean_squared_error(y_train, y_train_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_train, y_train_pred)

    valid_mse = mean_squared_error(y_valid, y_valid_pred)
    valid_rmse = np.sqrt(valid_mse)
    valid_mae = mean_absolute_error(y_valid, y_valid_pred)

    

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)

    mlflow.log_metric("valid mse", valid_mse)
    mlflow.log_metric("valid rmse", valid_rmse)
    mlflow.log_metric("valid mae", valid_mae)


    print(f"VALID MSE {valid_mse}")
    print(f"VALID RMSE: {valid_rmse}")
    print(f"VALID MAE: {valid_mae}")


    mlflow.sklearn.log_model(pipeline, model_name_tag)



r2_score(y_train, y_train_pred)
r2_score(y_valid, y_valid_pred)



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
from rfpimp import *  

rf_importances = importances(pipeline, X_train, y_train)
plot_importances(rf_importances.head(30))




# PERMUTATION IMPORTANCE -----------------------------------------------------------

from sklearn.inspection import permutation_importance
result = permutation_importance(pipeline, X_valid, y_valid, n_repeats=5, random_state=32)
importances = result.importances_mean



rel_cat_feats_low_card

pipeline.get_feature_names_out()

cat_feats = 
feats = [date_feats + rel_num_feats]





# GET ODDS DATA -------------------------------------
odds_data = get_odds_data()

odds_data

from shapash.explainer.smart_explainer import SmartExplainer
xpl = SmartExplainer(model=model,
    preprocessing=col_trans_pipeline) 


xpl.compile(x=X_valid,
            y_target=y_valid # Optional: allows to display True Values vs Predicted Values
           )



# I THINK THIS IS DONE LATER ------------------------------------
# training error versus number of train records?

# plot training error versus parameter complexity? --- this would be number of trees, or min_leaf_size for random forest and xgboost





# we'll save cross validation for the final steps here ------------------------

tscv = TimeSeriesSplit(n_splits=5)

cross_val_scores_r2 = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring=make_scorer(r2_score))
cross_val_score_mse = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
cross_val_score_mae = cross_val_score(pipeline, X_train, y_train, cv=tscv, scoring='neg_mean_absolute_error')

# mlflow.log_metric('cv_r2', cross_val_scores_r2.mean())
# mlflow.log_metric('cv_neg_mse', cross_val_score_mse.mean())
# mlflow.log_metric('cv_rmse', np.mean(np.sqrt(np.abs(cross_val_score_mse))))
# mlflow.log_metric('cv_mae', cross_val_score_mae.mean())




# APPENDIX -------------------------------------------------------------------------------

## num feats trend - general -----------
loop_place=0

for col in lagged_num_cols_complete:
    
    temp_lagged_col_df = pd.DataFrame()

    ## accuracy flatlines at about 5 games out, with each game out happening there after only reducing by 0.1
    lagged_col_label = f'team_lagged_{col}'
    temp_lagged_col_df[lagged_col_label] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])[col].transform(lambda x: x.shift(1))

    for i in range(2,6):

        for stat_type in ['mean', 'median', 'std']:

            col_label = f'team_lagged_{col}_rolling_{i}_{stat_type}'
            temp_lagged_col_df[col_label] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON'])[col].transform(lambda x: x.shift(1).rolling(i, min_periods=i).agg(stat_type))

    game_team_regular_train = pd.concat([game_team_regular_train, temp_lagged_col_df], axis=1)

    loop_place+=1

    pct_complate = loop_place/len(lagged_num_cols_complete)
    print("{:.2%}".format(pct_complate))

    del temp_lagged_col_df

# WE HAVE TO REMOVE THIS BECAUSE IT CREATES TOO MANY NULLS 


## num feats trend - home_visitor -----------------------
loop_place=0

home_visitor_stats_track = ['PTS', 'PTS_allowed_opposing']

temp_lagged_col_df = pd.DataFrame()

for col in home_visitor_stats_track:

    lagged_col_label = f'team_lagged_home_visitor_{col}'
    temp_lagged_col_df[lagged_col_label] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON', 'HOME_VISITOR'])[col].transform(lambda x: x.shift(1))


    for i in range(2,6):
        
        for stat_type in ['mean', 'median', 'std']:
            col_label = f'team_lagged_home_visitor_{col}_rolling_{i}_{stat_type}'
            temp_lagged_col_df[col_label] = game_team_regular_train.groupby(['TEAM_ID', 'SEASON', 'HOME_VISITOR'])[col].transform(lambda x: x.shift(1).rolling(i, min_periods=i).agg(stat_type))


    loop_place+=1

    pct_complate = loop_place/len(home_visitor_stats_track)
    print("{:.2%}".format(pct_complate))
    
    
game_team_regular_train = pd.concat([game_team_regular_train, temp_lagged_col_df], axis=1)

del temp_lagged_col_df

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