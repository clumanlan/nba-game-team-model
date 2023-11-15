from functools import wraps
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import awswrangler as wr
from time import time, sleep
from nba_api.stats.static import teams, players
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, ScoreboardV2, BoxScoreAdvancedV2, BoxScoreTraditionalV2
from nba_api.stats.library.parameters import SeasonAll
from nba_api.stats.static import players
import awswrangler as wr
from typing import List, Tuple

# GAME_ID: OBJECT
# TEAM_ID, PLAYER_ID: INT64
#

# create a timing decorator that you can pass to other functions
# decorators work so you're able to pass functions in perform a task then pass it back out really handy when they're
# same tasks

def timing(f):
    @wraps(f)
    def wrap(*arg,**kw):

        time_start = time()
        print(f"Starting function: {f.__name__}..........")

        result = f (*arg, **kw)
        time_end = time()
        
        print(f"Completed function: {f.__name__} in {round((time_end-time_start)/60,2)} minutes")

        return result
    return wrap
    

@timing
def get_game_ids_pulled() -> Tuple[pd.Series, str]:
    """
    Retrieves the unique game IDs and latest game date from game_stats data previously pulled.
    
    Parameters:
    s3_path (str): S3 path to the game_stats data.
    
    Returns:
    Tuple[pd.Series, str]: A tuple containing a pandas Series of unique game IDs pulled and the latest game date in the data.
    """

    s3_path = "s3://nbadk-model/game_stats"

    game_headers = wr.s3.read_parquet(
        path=s3_path,
        path_suffix = ".parquet" ,
        use_threads =True
    )

    game_ids_pulled = game_headers.GAME_ID.unique()
    latest_game_date = pd.to_datetime(game_headers.GAME_DATE_EST).dt.strftime('%Y-%m-%d').unique().max()
    
    print(f'latest game date pulled: {latest_game_date}')
    
    return game_ids_pulled, latest_game_date


@timing
def get_game_data(start_date:str) -> Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], List[date]]:
    """
    Fetches game data from ScoreboardV2 API for each date from `start_date` to today's date.
    :param start_date: A string representing the start date in the format YYYY-MM-DD.
    :return: A tuple containing a list of tuples of Game Header and Team Game Line Scores dataframes and a list of dates where an error occurred.
    """

    game_header_w_standings_list = []
    team_game_line_score_list = []
    error_dates_list = []

    start_date = datetime.strptime(start_date, '%Y-%m-%d').date() - timedelta(days=3)

    end_date = date.today()
    end_date_string = end_date.strftime('%Y-%m-%d')

    current_date = start_date

    while current_date <= end_date:
        try:
            scoreboard = ScoreboardV2(game_date=current_date, league_id='00')

            game_header = scoreboard.game_header.get_data_frame()
            print(f"    currently pulling {current_date}: {game_header.shape}")

            series_standings = scoreboard.series_standings.get_data_frame()
            series_standings.drop(['HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'GAME_DATE_EST'], axis=1, inplace=True)

            game_header_w_standings = game_header.merge(series_standings, on='GAME_ID')

            # each line rpresents a game-teamid
            team_game_line_score = scoreboard.line_score.get_data_frame()
            game_header_w_standings_list.append(game_header_w_standings)
            team_game_line_score_list.append(team_game_line_score)
        
        except Exception as e:
            error_dates_list.append(current_date)
            print(f'error {current_date}')

        current_date += timedelta(days=1)

        sleep(1.1)

    
    # sppecify data types explicitly
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

    game_header_w_standings_df = pd.concat(game_header_w_standings_list)
    team_game_line_score_df = pd.concat(team_game_line_score_list)


    game_header_w_standings_df = game_header_w_standings_df.astype(dtype=game_header_dtype_dict)
    team_game_line_score_df = team_game_line_score_df.astype(dtype=team_game_dtype_dict)


    return game_header_w_standings_df, team_game_line_score_df,  error_dates_list



@timing
def filter_and_write_game_data(game_header_w_standings_df: pd.DataFrame, team_game_line_score_df: pd.DataFrame, game_ids_pulled: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, List[int]]:
    """
    Filters the game data by removing games that are not yet completed and have already been processed.

    :param game_data: A list of tuples containing two dataframes representing game data.
    :param game_ids: A list of integers representing game IDs that have already been processed and stored in S3

    :return: A tuple containing two dataframes representing the filtered game data and a list of Game IDS to pull
    """

    game_header_w_standings_df_filtered = game_header_w_standings_df[(game_header_w_standings_df['LIVE_PERIOD'] >= 4) & (~game_header_w_standings_df['GAME_ID'].isin(game_ids_pulled))]
    team_game_line_score_df_filtered = team_game_line_score_df[team_game_line_score_df['GAME_ID'].isin(game_header_w_standings_df_filtered.GAME_ID)]

    game_ids = game_header_w_standings_df_filtered.GAME_ID.unique()

    print(f'    Game ids to pull {len(game_ids)}')

    if len(game_ids) == 0:
        pass

    else:
        print('    Writing Game Header data to S3...........')
        output_date = datetime.today().strftime('%Y-%m-%d')
        
        wr.s3.to_parquet(
            df=game_header_w_standings_df_filtered,
            path=f's3://nbadk-model/game_stats/game_header/rolling/game_header_{output_date}.parquet'
        )
        wr.s3.to_parquet(
            df=team_game_line_score_df_filtered,
            path=f's3://nbadk-model/team_stats/game_line_score/rolling/game_line_score_{output_date}.parquet'
        )


    return game_ids



@timing
def get_boxscore_traditional(game_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Retrieve traditional boxscore data for given game IDs.

    Parameters:
    -----------
    game_ids : list of str
        List of game IDs for which boxscore data is to be retrieved.

    Returns:
    --------
    Tuple of two dataframes: player-level boxscore data and team-level boxscore data.
    """


    boxscore_trad_player_list = []
    boxscore_trad_team_list = []
    boxscore_trad_error_list = []
    game_len = game_ids.shape[0]
    loop_place = 0

    for game_id in game_ids:
        try:
            boxscore_trad = BoxScoreTraditionalV2(game_id=game_id)

            boxscore_trad_player = boxscore_trad.player_stats.get_data_frame()
            boxscore_trad_team = boxscore_trad.team_stats.get_data_frame()

            boxscore_trad_player_list.append(boxscore_trad_player)
            boxscore_trad_team_list.append(boxscore_trad_team)

        
        except Exception as e:
            boxscore_trad_error_list.append(game_id)

            print(f'error {game_id}')
        
        sleep(1.1)
        loop_place += 1
        print(f'    {round((loop_place/game_len)*100,2)} % complete')

    boxscore_traditional_player_df = pd.concat(boxscore_trad_player_list)
    boxscore_traditional_team_df = pd.concat(boxscore_trad_team_list)

    return boxscore_traditional_player_df, boxscore_traditional_team_df





@timing
def write_boxscore_traditional_to_s3(boxscore_traditional_player_df: pd.DataFrame, boxscore_traditional_team_df: pd.DataFrame) -> None:
    """
    Writes boxscore traditional stats for players and teams to S3 in parquet format with today's date appended to the filename.

    Parameters:
        boxscore_traditional_player_df (pd.DataFrame): DataFrame of player boxscore traditional stats
        boxscore_traditional_team_df (pd.DataFrame): DataFrame of team boxscore traditional stats
    """
    today = date.today()
    today_string = today.strftime('%Y-%m-%d')

    
    boxscore_trad_player_dtype_mapping = {
        'GAME_ID': 'object',
        'TEAM_ID': 'int64',
        'TEAM_ABBREVIATION': 'object',
        'TEAM_CITY': 'object',
        'PLAYER_ID': 'int64',
        'PLAYER_NAME': 'object',
        'NICKNAME': 'object',
        'START_POSITION': 'object',
        'COMMENT': 'object',
        'MIN': 'object', 
        'FGM': 'float64',
        'FGA': 'float64',
        'FG_PCT': 'float64',
        'FG3M': 'float64',
        'FG3A': 'float64',
        'FG3_PCT': 'float64',
        'FTM': 'float64',
        'FTA': 'float64',
        'FT_PCT': 'float64',
        'OREB': 'float64',
        'DREB': 'float64',
        'REB': 'float64',
        'AST': 'float64',
        'STL': 'float64',
        'BLK': 'float64',
        'TO': 'float64',
        'PF': 'float64',
        'PTS': 'float64',
        'PLUS_MINUS': 'float64'
    }

    boxscore_trad_team_dtype_mapping = {
        'GAME_ID': 'object',
        'TEAM_ID': 'int64',
        'TEAM_NAME': 'object',
        'TEAM_ABBREVIATION': 'object',
        'TEAM_CITY': 'object',
        'MIN': 'object',
        'FGM': 'float64',
        'FGA': 'float64',
        'FG_PCT': 'float64',
        'FG3M': 'float64',
        'FG3A': 'float64',
        'FG3_PCT': 'float64',
        'FTM': 'float64',
        'FTA': 'float64',
        'FT_PCT': 'float64',
        'OREB': 'float64',
        'DREB': 'float64',
        'REB': 'float64',
        'AST': 'float64',
        'STL': 'float64',
        'BLK': 'float64',
        'TO': 'float64',
        'PF': 'float64',
        'PTS': 'float64',
        'PLUS_MINUS': 'float64'
    }


    boxscore_traditional_player_df = boxscore_traditional_player_df.astype(dtype=boxscore_trad_player_dtype_mapping)
    boxscore_traditional_team_df = boxscore_traditional_team_df.astype(dtype=boxscore_trad_team_dtype_mapping)

    print('     Writing Boxscore Traditional to S3......................')

    wr.s3.to_parquet(
            df=boxscore_traditional_player_df,
            path=f's3://nbadk-model/player_stats/boxscore_traditional/boxscore_traditional_player_{today_string}.parquet'
        )

    wr.s3.to_parquet(
        df=boxscore_traditional_team_df,
        path=f's3://nbadk-model/team_stats/boxscore_traditional/boxscore_traditional_team_{today_string}.parquet'
        )
    

    return None




@timing
def get_boxscore_advanced(game_ids:list) ->  Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Retrieves box score advanced statistics (e.g. PACE) for a list of game ids.

    Args:
        game_ids (List[str]): List of game ids to retrieve box score advanced statistics for.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[str]]: A tuple containing two dataframes - player_boxscore_advanced_stats_df
        and team_boxscore_stats_advanced_df

    """

    today = date.today()
    today_string = today.strftime('%Y-%m-%d')

    player_boxscore_stats_list = []
    team_boxscore_stats_list = []
    error_game_id_list = []

    game_len = len(game_ids)
    loop_place = 0

    for game_id in game_ids:
        print(f'    Starting {game_id}')

        try:
            boxscore_stats_adv = BoxScoreAdvancedV2(game_id=game_id)

            player_boxscore_stats = boxscore_stats_adv.player_stats.get_data_frame()
            team_boxscore_stats = boxscore_stats_adv.team_stats.get_data_frame()

            player_boxscore_stats_list.append(player_boxscore_stats)
            team_boxscore_stats_list.append(team_boxscore_stats)

        
        except Exception as e:
            error_game_id_list.append(game_id)

            print(f'    error {game_id}')
        
        loop_place += 1
        print(f'    {round((loop_place/game_len)*100,2)} % complete')
        sleep(1.1)
    
    player_boxscore_advanced_stats_df = pd.concat(player_boxscore_stats_list)
    team_boxscore_stats_advanced_df = pd.concat(team_boxscore_stats_list)

    return player_boxscore_advanced_stats_df, team_boxscore_stats_advanced_df




@timing
def write_boxscore_advanced_to_s3(player_boxscore_advanced_stats_df: pd.DataFrame, 
                                  team_boxscore_stats_advanced_df: pd.DataFrame) -> None:
    """
    Writes the boxscore advanced data to S3 in Parquet format.

    :param player_boxscore_advanced_stats_df: A dataframe representing the player boxscore advanced stats data.
    :param team_boxscore_stats_advanced_df: A dataframe representing the team boxscore advanced stats data.

        """
    today = date.today()
    today_string = today.strftime('%Y-%m-%d')

        
    player_boxscore_advanced_dtypes = {
        'GAME_ID': 'object',
        'TEAM_ID': 'int64',
        'TEAM_ABBREVIATION': 'object',
        'TEAM_CITY': 'object',
        'PLAYER_ID': 'int64',
        'PLAYER_NAME': 'object',
        'NICKNAME': 'object',
        'START_POSITION': 'object',
        'COMMENT': 'object',
        'MIN': 'object',
        'E_OFF_RATING': 'float64',
        'OFF_RATING': 'float64',
        'E_DEF_RATING': 'float64',
        'DEF_RATING': 'float64',
        'E_NET_RATING': 'float64',
        'NET_RATING': 'float64',
        'AST_PCT': 'float64',
        'AST_TOV': 'float64',
        'AST_RATIO': 'float64',
        'OREB_PCT': 'float64',
        'DREB_PCT': 'float64',
        'REB_PCT': 'float64',
        'TM_TOV_PCT': 'float64',
        'EFG_PCT': 'float64',
        'TS_PCT': 'float64',
        'USG_PCT': 'float64',
        'E_USG_PCT': 'float64',
        'E_PACE': 'float64',
        'PACE': 'float64',
        'PACE_PER40': 'float64',
        'POSS': 'float64',
        'PIE': 'float64'
    }

    team_boxscore_advanced_dtypes = {
        'GAME_ID': 'object',
        'TEAM_ID': 'int64',
        'TEAM_NAME': 'object',
        'TEAM_ABBREVIATION': 'object',
        'TEAM_CITY': 'object',
        'MIN': 'object',  
        'E_OFF_RATING': 'float64',
        'OFF_RATING': 'float64',
        'E_DEF_RATING': 'float64',
        'DEF_RATING': 'float64',
        'E_NET_RATING': 'float64',
        'NET_RATING': 'float64',
        'AST_PCT': 'float64',
        'AST_TOV': 'float64',
        'AST_RATIO': 'float64',
        'OREB_PCT': 'float64',
        'DREB_PCT': 'float64',
        'REB_PCT': 'float64',
        'E_TM_TOV_PCT': 'float64',
        'TM_TOV_PCT': 'float64',
        'EFG_PCT': 'float64',
        'TS_PCT': 'float64',
        'USG_PCT': 'float64',
        'E_USG_PCT': 'float64',
        'E_PACE': 'float64',
        'PACE': 'float64',
        'PACE_PER40': 'float64',
        'POSS': 'float64',
        'PIE': 'float64'
    }

    
    player_boxscore_advanced_stats_df = player_boxscore_advanced_stats_df.astype(dtype=player_boxscore_advanced_dtypes)
    team_boxscore_stats_advanced_df = team_boxscore_stats_advanced_df.astype(dtype=team_boxscore_advanced_dtypes)


    print('     Writing Boxscore Advanced to S3')

    wr.s3.to_parquet(
        df=player_boxscore_advanced_stats_df,
        path=f's3://nbadk-model/player_stats/boxscore_advanced/player_boxscore_advanced_stats_{today_string}.parquet'
    )

    wr.s3.to_parquet(
        df=team_boxscore_stats_advanced_df,
        path=f's3://nbadk-model/team_stats/boxscore_advanced/team_boxscore_advanced_stats_{today_string}.parquet'
    )

    return None



if __name__ == '__main__':

    game_ids_pulled, latest_game_pulled_date = get_game_ids_pulled()

    game_header_w_standings_df, team_game_line_score_df,  error_dates_list = get_game_data(latest_game_pulled_date)

    game_ids = filter_and_write_game_data(game_header_w_standings_df, team_game_line_score_df, game_ids_pulled)

    if len(game_ids) != 0:
            
        boxscore_traditional_player_df, boxscore_traditional_team_df = get_boxscore_traditional(game_ids)
        write_boxscore_traditional_to_s3(boxscore_traditional_player_df, boxscore_traditional_team_df)
    
        player_boxscore_advanced_stats_df, team_boxscore_stats_advanced_df = get_boxscore_advanced(game_ids)
        write_boxscore_advanced_to_s3(player_boxscore_advanced_stats_df, team_boxscore_stats_advanced_df)

    else: 
        print('No game IDs to be pulled')

