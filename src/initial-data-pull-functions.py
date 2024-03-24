
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
from time import sleep
import geopy.distance

# TO DO: 
#        CREATE A FUNCTION THAT LOOKS AT GAME IDS  IN A GIVEN DATAFRAME THEN CHECKS WHETHER WE ARE MISSING ANY IN GAME HEADER
# i only need to check one of the player or team since advanced and traditional are in each

# FUNCTION TO PULL AND CHECK MISSING GAME_IDS

game_header_last_check_date = '2024-01-06'
game_header_error_dates = ['2001-03-13', '2019-06-12']


def get_all_game_headers(start_date:str='2001-01-01'):
    """
    Fetches game data from ScoreboardV2 API for each date from `start_date` to today's date.
    :param start_date: A string representing the start date in the format YYYY-MM-DD.
    :return: A tuple containing a list of tuples of Game Header and Team Game Line Scores dataframes and a list of dates where an error occurred.
    """

    game_header_w_standings_list = []
    team_game_line_score_list = []
    error_dates_list = []

    end_date = date.today()
    end_date_string = end_date.strftime('%Y-%m-%d')

    current_date = dt.datetime.strptime(start_date, '%Y-%m-%d').date()

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

        current_date += dt.timedelta(days=1)

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



def get_game_header_game_ids():

    game_stats_path = "s3://nbadk-model/game_stats/game_header"

    game_headers_df = wr.s3.read_parquet(
        path=game_stats_path,
        path_suffix = ".parquet" ,
        use_threads =True
    )

    game_ids_pulled = game_headers_df['GAME_ID'].unique()

    return game_ids_pulled


def get_game_headers():

    game_stats_path = "s3://nbadk-model/game_stats/game_header"

    game_headers_df = wr.s3.read_parquet(
        path=game_stats_path,
        path_suffix = ".parquet" ,
        use_threads =True
    )

    return game_headers_df



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




# GET BOXSCORE TRADITIONAL STATS -------------------------------------------------

def pull_boxscore_traditional(game_ids, df_label=dt.datetime.now().strftime("%Y-%m-%d")):

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

            print(game_id)
        
        except Exception as e:
            boxscore_trad_error_list.append(game_id)

            print(f'error {game_id}')
        
        time.sleep(.60)
        loop_place += 1
        print(f'{round((loop_place/game_len),3)*100} % complete')

    boxscore_traditional_player_df = pd.concat(boxscore_trad_player_list)
    boxscore_traditional_team_df = pd.concat(boxscore_trad_team_list)
    
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


    print('Writing Boxscore Traditional to S3.....................')

    wr.s3.to_parquet(
            df=boxscore_traditional_player_df,
            path=f"s3://nbadk-model/player_stats/boxscore_traditional/updated/boxscore_traditional_player_{df_label}.parquet"
        )

    wr.s3.to_parquet(
        df=boxscore_traditional_team_df,
        path=f"s3://nbadk-model/team_stats/boxscore_traditional/updated/boxscore_traditional_team_{df_label}.parquet"
        )

    return boxscore_trad_error_list







# need to change this function so it has the mapping of data types 


def pull_boxscore_advanced(game_ids, df_label=dt.datetime.now().strftime("%Y-%m-%d")):

    player_boxscore_stats_list = []
    team_boxscore_stats_list = []
    error_game_id_list = []

    game_len = len(game_ids)
    loop_place = 0

    for game_id in game_ids:
        print(f'Starting {game_id}')

        try:
            boxscore_stats_adv = BoxScoreAdvancedV2(game_id=game_id)

            player_boxscore_stats = boxscore_stats_adv.player_stats.get_data_frame()
            team_boxscore_stats = boxscore_stats_adv.team_stats.get_data_frame()

            player_boxscore_stats_list.append(player_boxscore_stats)
            team_boxscore_stats_list.append(team_boxscore_stats)

            print(f'success {game_id}')
        
        except Exception as e:
            error_game_id_list.append(game_id)

            print(f'error {game_id}')
        
        loop_place += 1
        print(f'{round(loop_place/game_len,3)*100} % complete')
        time.sleep(1)

    player_boxscore_advanced_stats_df = pd.concat(player_boxscore_stats_list)
    team_boxscore_stats_advanced_df = pd.concat(team_boxscore_stats_list)
    
    player_ids = player_boxscore_advanced_stats_df.PLAYER_ID.unique()
    player_ids_df = pd.DataFrame(player_ids, columns=['player_id'])


        
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


    print('Writing Boxscore Advanced to S3.....................')


    wr.s3.to_parquet(
        df=player_boxscore_advanced_stats_df,
        path=f"s3://nbadk-model/player_stats/boxscore_advanced/updated/player_boxscore_advanced_stats_{df_label}.parquet"
    )

    wr.s3.to_parquet(
        df=team_boxscore_stats_advanced_df,
        path=f"s3://nbadk-model/team_stats/boxscore_advanced/updated/team_boxscore_advanced_stats_{df_label}.parquet"
    )

    return error_game_id_list


error_adv_game_ids = pull_boxscore_advanced(['0020300778'])


# GET AND WRITE ODDS DATA  -------------------------------------------------

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



# STADIUM DISTANCE  ---------------------------------------------------------------------------

def calculate_distance(row):
    coords_1 = (row['Latitude_a'], row['Longitude_a'])
    coords_2 = (row['Latitude_b'], row['Longitude_b'])

    return geopy.distance.geodesic(coords_1, coords_2).miles

def create_stadium_distance_df():

    stadium_locations = {
        'Team': ['Los Angeles Lakers', 'Los Angeles Clippers', 'New York Knicks', 'Golden State Warriors', 'Milwaukee Bucks',
                'Dallas Mavericks', 'Boston Celtics', 'Chicago Bulls', 'Toronto Raptors', 'Cleveland Cavaliers',
                'New Orleans Pelicans', 'Philadelphia 76ers', 'Houston Rockets', 'Denver Nuggets',
                'Atlanta Hawks', 'Miami Heat', 'Utah Jazz', 'Minnesota Timberwolves', 'Indiana Pacers',
                'Portland Trail Blazers', 'Charlotte Hornets', 'Sacramento Kings', 'Detroit Pistons',
                'Orlando Magic', 'Phoenix Suns', 'Brooklyn Nets', 'San Antonio Spurs', 'Oklahoma City Thunder',
                'Washington Wizards', 'Memphis Grizzlies'],
        'Latitude': [34.043056, 34.043056, 40.750556, 37.768056, 43.043611, 32.790556, 42.366389, 41.880556, 43.643333, 41.496944,
                    29.949722, 39.952778, 29.750833, 39.748611, 33.757222, 25.781389, 40.768333, 44.979444, 39.763889,
                    45.531667, 35.205833, 38.580833, 38.751667, 42.341111, 28.539167, 33.445833, 40.68265, 29.426944,
                    38.898056, 35.138333],
        'Longitude': [-118.267222, -118.267222, -73.993611, -122.3875, -87.916944, -96.810278, -71.062222, -87.674167, -79.379167, -81.688889,
                    -90.081944, -75.190833, -95.370833, -104.996389, -84.396389, -80.188611, -111.901111, -93.276111, -86.155556,
                    -122.666389, -80.839167, -121.968611, -77.012222, -83.045833, -81.379722, -112.2625, -73.974689, -98.495833,
                    -95.341944, -77.036944],
    }


    nba_stadiums_df = pd.DataFrame(stadium_locations)

    nba_stadiums_df = (
        nba_stadiums_df
        .assign(TEAM_NAME= nba_stadiums_df['Team'].str.split().apply(lambda x: x[-1]),
                _key = 1) # we use this key to cross join
        .drop('Team', axis=1)
    )

    nba_stadiums_df = pd.merge(nba_stadiums_df, nba_stadiums_df, suffixes=['_a', '_b'], on='_key').drop('_key', axis=1)
    nba_stadiums_df = nba_stadiums_df[nba_stadiums_df['TEAM_NAME_a']!=nba_stadiums_df['TEAM_NAME_b']].reset_index(drop=True)

    nba_stadiums_df['distance_miles'] = nba_stadiums_df.apply(calculate_distance, axis=1)


    wr.s3.to_parquet(
        df=nba_stadiums_df,
        path=f's3://nbadk-model/stadiuminfo/stadium_distances/nba_stadium_location_distances.parquet'
    )



game_ids_pulled = get_game_header_game_ids()

game_ids_pulled = set(game_ids_pulled)

player_info_df, boxscore_trad_player_df, boxscore_adv_player_df = get_player_dfs()

boxscore_trad_team_df, boxscore_adv_team_df = get_team_level_dfs()

# start in 2004?

# there are no advanced stats for preseason and 2011 is a shortened season

