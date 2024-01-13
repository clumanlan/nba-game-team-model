import pandas as pd
import sklearn 
import seaborn as sns
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, ScoreboardV2, BoxScoreAdvancedV2, BoxScoreTraditionalV2
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import players
from datetime import date
import awswrangler as wr
import numpy as np
import time
import datetime as dt
import mlflow

# TODO: validate data :) create a baseline model, setup infrastrcutre for CI/CD to SAGEMAKER WITH MLFLOW
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

rel_game_ids = game_headers_filtered.GAME_ID.unique()


game_headers_pivot = pd.melt(game_headers_filtered, id_vars=['GAME_ID','game_type', 'SEASON', 'GAME_DATE_EST'], value_vars=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'], var_name='HOME_VISITOR', value_name='TEAM_ID')
game_headers_pivot['HOME_VISITOR'] = game_headers_pivot['HOME_VISITOR'].replace({'HOME_TEAM_ID':'HOME', 'VISITOR_TEAM_ID':'VISITOR'})



boxscore_trad_team_df, boxscore_adv_team_df = get_team_level_dfs()

bxscore_trad_irrelevant_cols = ['TEAM_ABBREVIATION', 'TEAM_CITY']
boxscore_trad_team_df_filtered = boxscore_trad_team_df.drop(bxscore_trad_irrelevant_cols, axis=1)

adv_team_team_identity_cols = ['TEAM_NAME', 'TEAM_ABBREVIATION', 'TEAM_CITY','MIN']
boxscore_adv_team_df_filtered = boxscore_adv_team_df.drop(adv_team_team_identity_cols, axis=1)



# Setup Mlflow ------------------------------------------------
## mlflow server --backend-store-uri sqlite:///mlflow.db 
## --default-artifact-root mlruns/ 
experiment_name = "team_game"
#mlflow.create_experiment(experiment_name, artifact_location="s3://nbadk-model/models/team-game/experiments")
mlflow.set_experiment(experiment_name)










# what to name 
A.set_index('key').join([B2, C2], how='inner').reset_index()



boxscore_combined_trad = game_headers_df_processed_filtered.merge(boxscore_trad_team_df, left_on=['GAME_ID', 'HOME_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'], how='left')

boxscore_combined_adv = game_headers_df_processed_filtered.merge(boxscore_adv_team_df, left_on=['GAME_ID', 'HOME_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'], how='left')

boxscore_combined_trad[boxscore_combined_trad['PTS'].isnull()]['SEASON'].value_counts()

missing_game_ids_trad = boxscore_combined_trad[boxscore_combined_trad['PTS'].isnull()]['GAME_ID'].unique()
missing_game_df = game_headers_df_processed_filtered[game_headers_df_processed_filtered['GAME_ID'].isin(missing_game_ids_trad)]


missing_game_ids_adv = boxscore_combined_adv[boxscore_combined_adv['PIE'].isnull()]['GAME_ID'].unique()

len(missing_game_ids_trad)





len(missing_game_ids_adv)


boxscore_adv_error = get_boxscore_advanced(missing_game_ids_adv)
boxscore_trad_error = get_boxscore_traditional(missing_game_ids_trad)

boxscore_adv_error_remaining = get_boxscore_advanced(boxscore_adv_error)






get_player_dfs()

# need to check game IDS 



# CHECK MISSING FUNCTIONS ---------------------------------------------


trad_game_id_missing = [game_id for game_id in boxscore_trad_team_df.GAME_ID.unique() if game_id not in game_header_game_ids_complete]
adv_game_id_missing = [game_id for game_id in boxscore_adv_team_df.GAME_ID.unique() if game_id not in game_header_game_ids_complete]


adv_game_id_missing

trad_missing_from_adv = [game_id for game_id in boxscore_trad_team_df.GAME_ID.unique() if game_id not in  boxscore_adv_team_df.GAME_ID.unique()]




boxscore_adv_team_df = boxscore_adv_team_df.drop(['TEAM_NAME', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'MIN'], axis=1)

team_boxscore_combined = pd.merge(boxscore_trad_team_df, boxscore_adv_team_df, on=['GAME_ID', 'TEAM_ID'], how='outer')


