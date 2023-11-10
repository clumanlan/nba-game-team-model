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
# FUNCTIONS ---------------------------------------


def get_game_headers() -> tuple:

    game_stats_path = "s3://nbadk-model/game_stats"

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


# GET ODDS DATA!!!!

odds_shark_team_id = {20722: 'celtics', 20749: 'nets', 20747: 'knicks', 20731: '76ers',
                      20742: 'raptors', 20732: 'bulls', 20735: 'cavaliers', 20743: 'pistons',
                      20737: 'pacers', 20725: 'bucks', 20734: 'hawks', 20751: 'hornets', 
                      20726: 'heat', 20750: 'magic', 20746: 'wizards',  20723: 'nuggets',
                      20744: 'timeberwolves', 20728: 'thunder', 20748: 'trail blazers', 
                      20738: 'jazz', 20741: 'warriors', 20736: 'clippers', 20739: 'lakers',
                      20730: 'suns', 20745: 'kings', 20727: 'mavericks', 20740: 'rockets',
                      20729: 'grizzlies', 20733: 'pelicans', 20724: 'spurs'}


odds_list = []

for team_id in odds_shark_team_id:

    print(team_id)

    for season in range(2020,2025):
        print(season)
        try:
            odds_df = pd.read_html(f'https://www.oddsshark.com/stats/gamelog/basketball/nba/{team_id}?season={season}')[0]
            odds_df['team_id'] = team_id
            odds_df['season'] = season

            odds_list.append(odds_df)
            time.sleep(1.1)
        except: 
            print(f'error on team:{team_id} and season: {season}')



complete_odds_df = pd.concat(odds_list).reset_index(drop=True)

complete_odds_df['team_name'] = complete_odds_df['team_id'].map(odds_shark_team_id)


wr.s3.to_parquet(
    df=complete_odds_df,
    path=f's3://nbadk-model/oddsshark/team_odds/nba_team_odds_historical.parquet'
)


boxscore_trad_team_df, boxscore_adv_team_df = get_team_level_dfs()

game_headers_df_processed_filtered, game_home_away = get_game_headers()



boxscore_trad_team_df.columns

boxscore_adv_team_df.columns

