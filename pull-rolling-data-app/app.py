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
    
    print('latest game date pulled: {latest_game_date}')
    
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
            path=f's3://nbadk-model/game_stats/game_header/game_header_{output_date}.parquet'
        )
        wr.s3.to_parquet(
            df=team_game_line_score_df_filtered,
            path=f's3://nbadk-model/team_stats/game_line_score/game_line_score_{output_date}.parquet'
        )


    return  game_ids




game_ids_pulled, latest_game_pulled_date = get_game_ids_pulled()
game_header_w_standings_df, team_game_line_score_df,  error_dates_list = get_game_data(latest_game_pulled_date)
game_ids = filter_and_write_game_data(game_header_w_standings_df, team_game_line_score_df, game_ids_pulled)

