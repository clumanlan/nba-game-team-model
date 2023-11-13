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
# FUNCTIONS ---------------------------------------

game_line_score_path = "s3://nbadk-model/game_stats/game_line_score/game_line_score_initial_2023_11_10"

game_headers_df = wr.s3.read_parquet(
    path=game_stats_path,
    path_suffix = ".parquet" ,
    use_threads =True
)
game_header_w_standings_df = game_headers_df



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


# GET AND WRITE ODDS DATA 

def get_and_write_odds_data():
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


odds_path = "s3://nbadk-model/oddsshark/team_odds/nba_team_odds_historical.parquet"
odds_data = wr.s3.read_parquet(
        path=odds_path,
        path_suffix=".parquet",
        use_threads=True
    )



game_header_w_standings_list = []
team_game_line_score_list = []
error_dates_list = [pd.to_datetime('2011-06-06 00:00:00'), pd.to_datetime('2011-06-07 00:00:00')]


for date in error_dates_list:
    try:
        print(date.strftime('%Y-%m-%d'))
        scoreboard = ScoreboardV2(game_date=date.strftime('%Y-%m-%d'), league_id='00')

        game_header = scoreboard.game_header.get_data_frame()

        if game_header.shape[0]==0: # if no rows in game header then skip this date
            time.sleep(1.1)
            pass

        else:
            series_standings = scoreboard.series_standings.get_data_frame()
            series_standings.drop(['HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_DATE_EST'], axis=1, inplace=True)

            game_header_w_standings = game_header.merge(series_standings, on='GAME_ID')

            # each line rpresents a game-teamid
            team_game_line_score = scoreboard.line_score.get_data_frame()
            print(team_game_line_score.shape)
            game_header_w_standings_list.append(game_header_w_standings)
            team_game_line_score_list.append(team_game_line_score)
    
    except Exception as e:
        error_dates_list.append(date)
        print(f'error {date}')

    time.sleep(1.1)

    print(date)






game_header_w_standings_df = pd.concat(game_header_w_standings_list)
team_game_line_score_df = pd.concat(team_game_line_score_list)





# Specify data types explicitly
game_header_dtype_dict = {
    'GAME_DATE_EST': 'datetime64[ns]',
    'GAME_SEQUENCE': 'int64',
    'GAME_ID': 'object',
    'GAME_STATUS_ID': 'int64',
    'GAME_STATUS_TEXT': 'object',
    'GAMECODE': 'object',
    'HOME_TEAM_ID': 'int64',
    'VISITOR_TEAM_ID': 'int64',
    'SEASON': 'int64',
    'LIVE_PERIOD': 'int64',
    'LIVE_PC_TIME': 'object',
    'NATL_TV_BROADCASTER_ABBREVIATION': 'object',
    'HOME_TV_BROADCASTER_ABBREVIATION': 'object',
    'AWAY_TV_BROADCASTER_ABBREVIATION': 'object',
    'LIVE_PERIOD_TIME_BCAST': 'object',
    'ARENA_NAME': 'object',
    'WH_STATUS': 'bool',
    'WNBA_COMMISSIONER_FLAG': 'bool',
    'HOME_TEAM_WINS': 'int64',
    'HOME_TEAM_LOSSES': 'int64',
    'SERIES_LEADER': 'object'
}


# apply specify types for game header
game_header_w_standings_df = game_header_w_standings_df.astype(dtype=game_header_dtype_dict)


team_game_dtype_dict = {
    'GAME_DATE_EST': 'datetime64[ns]',
    'GAME_SEQUENCE': 'int64',
    'GAME_ID': 'object',
    'TEAM_ID': 'int64',
    'TEAM_ABBREVIATION': 'object',
    'TEAM_CITY_NAME': 'object',
    'TEAM_NAME': 'object',
    'TEAM_WINS_LOSSES': 'object',
    'PTS_QTR1': 'float64',
    'PTS_QTR2': 'float64',
    'PTS_QTR3': 'float64',
    'PTS_QTR4': 'float64',
    'PTS_OT1': 'float64',
    'PTS_OT2': 'float64',
    'PTS_OT3': 'float64',
    'PTS_OT4': 'float64',
    'PTS_OT5': 'float64',
    'PTS_OT6': 'float64',
    'PTS_OT7': 'float64',
    'PTS_OT8': 'float64',
    'PTS_OT9': 'float64',
    'PTS_OT10': 'float64',
    'PTS': 'float64',
    'FG_PCT': 'float64',
    'FT_PCT': 'float64',
    'FG3_PCT': 'float64',
    'AST': 'float64',
    'REB': 'float64',
    'TOV': 'float64'
}


team_game_line_score_df = team_game_line_score_df.astype(dtype=team_game_dtype_dict)


wr.s3.to_parquet(
    df=game_header_w_standings_df,
    path=f's3://nbadk-model/game_stats/game_header/initial/game_header_initial_2023_11_10.parquet'
)

# games that are null are postpoend 
wr.s3.to_parquet(
df=team_game_line_score_df,
path=f's3://nbadk-model/team_stats/game_line_score/game_line_score_initial_2023_11_10_missing_date.parquet'
)





team_game_line_score_df[team_game_line_score_df.PTS_QTR1.isnull()]


team_game_line_score_df.GAME_ID.value_counts(sort=True)
team_game_line_score_df.columns
game_header_w_standings_df.GAME_STATUS_ID.value_counts(sort=True)


game_headers_df_processed_filtered, game_home_away = get_game_headers()

game_header_game_ids_complete = game_headers_df_processed_filtered.GAME_ID.unique()


boxscore_trad_team_df, boxscore_adv_team_df = get_team_level_dfs()

trad_game_id_missing = [game_id for game_id in boxscore_trad_team_df.GAME_ID.unique() if game_id not in game_header_game_ids_complete]
adv_game_id_missing = [game_id for game_id in boxscore_adv_team_df.GAME_ID.unique() if game_id not in game_header_game_ids_complete]


adv_game_id_missing

trad_missing_from_adv = [game_id for game_id in boxscore_trad_team_df.GAME_ID.unique() if game_id not in  boxscore_adv_team_df.GAME_ID.unique()]




boxscore_adv_team_df = boxscore_adv_team_df.drop(['TEAM_NAME', 'TEAM_ABBREVIATION', 'TEAM_CITY', 'MIN'], axis=1)

team_boxscore_combined = pd.merge(boxscore_trad_team_df, boxscore_adv_team_df, on=['GAME_ID', 'TEAM_ID'], how='outer')
