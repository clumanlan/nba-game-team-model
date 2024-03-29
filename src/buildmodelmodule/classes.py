import awswrangler as wr
import pandas as pd
import numpy as np
from datetime import timedelta


# CREATE A CLASS THAT GETS PROCSSING TIME OF EACH OF THE FUNCTIONS AND SAYS WHEN THEY'RE STARTING 


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


class GetData():

    def __init__(self) -> None:
        pass

    def get_game_headers_historical(self) -> tuple:

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


        return game_headers_df_processed_filtered


    def get_player_dfs(self) -> tuple:
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


    def get_team_level_dfs(self) -> tuple:
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


    def get_odds_data(self):
        odds_path = "s3://nbadk-model/odds/oddsshark/team_odds/nba_team_odds_historical.parquet"

        odds_data = wr.s3.read_parquet(
                path=odds_path,
                path_suffix=".parquet",
                use_threads=True
            )

        return odds_data

    def get_odds_data_kaggle(self):
        odds_path = "s3://nbadk-model/odds/kaggle_historical/nba_betting_spread.csv"

        odds_data = wr.s3.read_csv(
                path=odds_path,
                path_suffix=".csv",
                use_threads=True
            )

        return odds_data
    
    def get_nba_stadiums_distances_df(self):
        
        nba_stadiums_distances_path = "s3://nbadk-model/stadiuminfo/stadium_distances/"

        nba_stadiums_distances_df = wr.s3.read_parquet(
            path=nba_stadiums_distances_path,
            path_suffix = ".parquet" ,
            use_threads =True
        )

        return nba_stadiums_distances_df



class TransformData():

    def __init__(self) -> None:

        self.static_cols = ['SEASON', 'TEAM_ID', 'TEAM_NAME', 'GAME_DATE_EST', 'GAME_ID']

        self.cat_cols = ['HOME_VISITOR', 'HOME', 'AWAY']

        self.lagged_num_cols = ['PTS', 'SEC', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
            'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
            'BLK', 'TO', 'PF',  'PLUS_MINUS', 'E_OFF_RATING', 'OFF_RATING',
            'E_DEF_RATING', 'DEF_RATING', 'E_NET_RATING', 'NET_RATING', 'AST_PCT',
            'AST_TOV', 'AST_RATIO', 'E_TM_TOV_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'USG_PCT',
            'E_USG_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE'] #  contains_sig_nulls = ['OREB_PCT', 'DREB_PCT', 'REB_PCT']


        self.lagged_num_cols_opposing = [f"{col}_allowed_opposing" for col in self.lagged_num_cols]
        
        self.lagged_num_cols_complete = self.lagged_num_cols + self.lagged_num_cols_opposing
        
        pass

    # CREATE GAME BOXSCORE AND OPPOSING STATS 

    def create_reg_season_game_boxscore(self, game_headers, boxscore_trad_team_df, boxscore_adv_team_df):

        game_headers_filtered = (
            game_headers[game_headers['game_type'].isin(['Regular Season', 'Post Season', 'Play-In Tournament'])]
            .drop(['HOME_TEAM_WINS', 'HOME_TEAM_LOSSES'], axis=1)
        )

        game_headers_filtered = pd.melt(game_headers_filtered, id_vars=['GAME_ID','game_type', 'SEASON', 'GAME_DATE_EST'], value_vars=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'], var_name='HOME_VISITOR', value_name='TEAM_ID')
        game_headers_filtered['HOME_VISITOR'] = game_headers_filtered['HOME_VISITOR'].replace({'HOME_TEAM_ID':'HOME', 'VISITOR_TEAM_ID':'VISITOR'})


        trad_cols_to_remove = ['TEAM_ABBREVIATION', 'TEAM_CITY']
        boxscore_trad_team_df = boxscore_trad_team_df.drop(trad_cols_to_remove, axis=1)

        adv_team_cols_to_remove = ['TEAM_NAME', 'TEAM_ABBREVIATION', 'TEAM_CITY','MIN']
        boxscore_adv_team_df = boxscore_adv_team_df.drop(adv_team_cols_to_remove, axis=1)

        team_boxscore_combined = pd.merge(boxscore_trad_team_df, boxscore_adv_team_df, on=['GAME_ID', 'TEAM_ID'], how='left')

        game_ids_missing = ['0020300778', '0022300023'] # check these later
        team_boxscore_combined = team_boxscore_combined[~team_boxscore_combined['GAME_ID'].isin(game_ids_missing)] # one game id missing trad & adv stats

        game_header_team_boxscore_combined = pd.merge(game_headers_filtered, team_boxscore_combined,  on=['GAME_ID', 'TEAM_ID'], how='inner')

        game_team_regular_season = (
            game_header_team_boxscore_combined[game_header_team_boxscore_combined['game_type']=='Regular Season']
            .assign(
                TEAM_ID = lambda x: x['TEAM_ID'].astype(str),
                SEC = lambda x: x['MIN'].apply(get_sec)
            )
            .sort_values(['GAME_DATE_EST', 'TEAM_ID'])
            .drop(['MIN'], axis=1)
            .reset_index(drop=True)
        )


        # ADD OPPOSING TEAM STATS (DEFENSIVE PERFORMANCE) -------------------------------------------------------------------------

        gr_train_home_base = game_team_regular_season[game_team_regular_season['HOME_VISITOR']=='HOME'][['TEAM_ID', 'GAME_ID']]
        gr_train_away_base = game_team_regular_season[game_team_regular_season['HOME_VISITOR']=='VISITOR'][['TEAM_ID', 'GAME_ID']]

        non_rel_opposing_cols = ['game_type', 'SEASON', 'GAME_DATE_EST', 'HOME_VISITOR', 'TEAM_ID', 'TEAM_NAME', 'OREB_PCT',
                                'DREB_PCT', 'REB_PCT']

        gr_train_home_stats = game_team_regular_season[game_team_regular_season['HOME_VISITOR']=='HOME'].drop(non_rel_opposing_cols, axis=1)
        gr_train_away_stats = game_team_regular_season[game_team_regular_season['HOME_VISITOR']=='VISITOR'].drop(non_rel_opposing_cols, axis=1)


        gr_train_home = pd.merge(gr_train_home_base, gr_train_away_stats, on='GAME_ID')
        gr_train_home.columns = [f"{col}_allowed_opposing" for col in gr_train_home.columns]
        gr_train_home = gr_train_home.rename({'TEAM_ID_allowed_opposing': 'TEAM_ID', 'GAME_ID_allowed_opposing': 'GAME_ID'}, axis=1)

        gr_train_away = pd.merge(gr_train_away_base, gr_train_home_stats, on='GAME_ID')
        gr_train_away.columns = [f"{col}_allowed_opposing" for col in gr_train_away.columns]
        gr_train_away = gr_train_away.rename({'TEAM_ID_allowed_opposing': 'TEAM_ID', 'GAME_ID_allowed_opposing': 'GAME_ID'}, axis=1)

        gr_train_opposing = pd.concat([gr_train_away, gr_train_home])

        game_team_regular_season = pd.merge(game_team_regular_season, gr_train_opposing, how='left', on=['TEAM_ID', 'GAME_ID'])

        return game_team_regular_season
    

    def process_regular_team_boxscore(self, game_team_regular_season):
        
        game_team_regular_season_processed = (
            game_team_regular_season
            .assign(
                SEASON = lambda x: x['SEASON'].astype(str),
                TEAM_ID_SEASON = lambda x: x['TEAM_ID'] + '_' + x['SEASON'],
                GAME_DATE_WEEK_START = lambda x: x['GAME_DATE_EST'].apply(lambda y: y - timedelta(days=y.isoweekday() % 7)),
                HOME = lambda x: np.where(x['HOME_VISITOR']=='HOME', 1, 0),
                AWAY = lambda x: np.where(x['HOME_VISITOR']=='VISITOR', 1, 0)
            )
        )

        rel_cols = self.static_cols + self.cat_cols + self.lagged_num_cols_complete 

        game_team_regular_season_processed = game_team_regular_season_processed[rel_cols]

        return game_team_regular_season_processed
    
    def add_visitor_stadium_distance(self, game_team_regular_season, stadium_distances_df):
        
        game_team_visitor_base = game_team_regular_season[['GAME_ID', 'TEAM_NAME', 'HOME_VISITOR']]

        game_team_visitor_base = game_team_visitor_base.merge(game_team_visitor_base, on='GAME_ID')
        game_team_visitor_base = game_team_visitor_base[(game_team_visitor_base['TEAM_NAME_x'] != game_team_visitor_base['TEAM_NAME_y']) & (game_team_visitor_base['HOME_VISITOR_x']=='VISITOR')]


        game_team_visitor_base = pd.merge(game_team_visitor_base, stadium_distances_df, left_on=['TEAM_NAME_x','TEAM_NAME_y'], right_on=['TEAM_NAME_a', 'TEAM_NAME_b'])

        game_team_visitor_base = game_team_visitor_base[['GAME_ID', 'distance_miles']]
        game_team_visitor_base['HOME_VISITOR'] = 'VISITOR'

        game_team_regular_season = game_team_regular_season.merge(game_team_visitor_base, on=['GAME_ID', 'HOME_VISITOR'], how='left')
        game_team_regular_season['distance_miles'] = game_team_regular_season['distance_miles'].fillna(0)

        return game_team_regular_season

            







# TIME TREND: WE CREATE TWO TYPES: SHORT AND LONG FOR ANY FEATURES WE CREATE ---------------------------------------------

class CreateTimeBasedFeatures():


    def __init__(self) -> None:

        self.static_cols = ['SEASON', 'TEAM_ID', 'TEAM_NAME', 'GAME_DATE_EST', 'GAME_ID']

        self.cat_cols = ['HOME', 'AWAY']

        self.lagged_num_cols = ['PTS', 'SEC', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
            'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL',
            'BLK', 'TO', 'PF',  'PLUS_MINUS', 'E_OFF_RATING', 'OFF_RATING',
            'E_DEF_RATING', 'DEF_RATING', 'E_NET_RATING', 'NET_RATING', 'AST_PCT',
            'AST_TOV', 'AST_RATIO', 'E_TM_TOV_PCT', 'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'USG_PCT',
            'E_USG_PCT', 'E_PACE', 'PACE', 'PACE_PER40', 'POSS', 'PIE'] #  contains_sig_nulls = ['OREB_PCT', 'DREB_PCT', 'REB_PCT']


        self.lagged_num_cols_opposing = [f"{col}_allowed_opposing" for col in self.lagged_num_cols]
        
        self.lagged_num_cols_complete = self.lagged_num_cols + self.lagged_num_cols_opposing

        pass



    def create_cat_short_trend(self, game_team_regular_season):

        game_team_regular_season = game_team_regular_season.set_index('GAME_DATE_EST')

        window_size = ['7D', '14D', '30D']

        loop_place = 0
            
        temp_lagged_cat_col_df = pd.DataFrame()

        for window in window_size:

            temp_lagged_cat_col_df[f'GAME_count_{window}'] = game_team_regular_season.groupby(['TEAM_ID', 'SEASON'])['GAME_ID'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).count())


            temp_lagged_cat_col_df[f'HOME_GAME_count_{window}'] = game_team_regular_season.groupby(['TEAM_ID', 'SEASON'])['HOME'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())


            temp_lagged_cat_col_df[f'AWAY_GAME_count_{window}'] = game_team_regular_season.groupby(['TEAM_ID', 'SEASON'])['AWAY'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())

            
        temp_lagged_cat_col_df = temp_lagged_cat_col_df.fillna(0)

        game_team_regular_season = pd.concat([game_team_regular_season,temp_lagged_cat_col_df], axis=1).reset_index()

        return game_team_regular_season
    
    

    def create_cat_season_trend(self, game_team_regular_season):

        temp_lagged_cat_col_df = pd.DataFrame()

        temp_lagged_cat_col_df['days_since_last_game'] = game_team_regular_season.groupby(['TEAM_ID', 'SEASON'])['GAME_DATE_EST'].diff().dt.days

        temp_lagged_cat_col_df[f'GAME_count_season'] = game_team_regular_season.groupby(['TEAM_ID', 'SEASON'])['GAME_ID'].transform(lambda x: x.shift(1).rolling(window=100, min_periods=1).count())

        temp_lagged_cat_col_df[f'HOME_GAME_count_season'] = game_team_regular_season.groupby(['TEAM_ID', 'SEASON'])['HOME'].transform(lambda x: x.shift(1).rolling(window=100, min_periods=1).sum())

        temp_lagged_cat_col_df[f'AWAY_GAME_count_season'] = game_team_regular_season.groupby(['TEAM_ID', 'SEASON'])['AWAY'].transform(lambda x: x.shift(1).rolling(window=100, min_periods=1).sum())

        temp_lagged_cat_col_df = temp_lagged_cat_col_df.fillna(0)

        game_team_regular_season = pd.concat([game_team_regular_season,temp_lagged_cat_col_df], axis=1)

        return game_team_regular_season



    # could rewrite this depending on what i eventaully try to do!!!
    def create_num_season_trend(self, game_team_regular_season):
        
        for col in self.lagged_num_cols_complete:

            temp_lagged_col_df = pd.DataFrame()

            for stat_type in ['mean', 'median', 'std']:
                
                col_label = f'team_lagged_{col}_rolling_season_{stat_type}'
                temp_lagged_col_df[col_label] = game_team_regular_season.groupby(['TEAM_ID', 'SEASON'])[col].transform(lambda x: x.shift(1).rolling(100, min_periods=1).agg(stat_type))
            

            game_team_regular_season = pd.concat([game_team_regular_season, temp_lagged_col_df], axis=1)


        del temp_lagged_col_df


        home_visitor_stats_track = ['PTS', 'PTS_allowed_opposing']

        temp_lagged_col_df = pd.DataFrame()

        for col in home_visitor_stats_track:

            for stat_type in ['mean', 'median', 'max', 'min']:

                col_label = f'team_lagged_home_visitor_{col}_rolling_season_{stat_type}'
                temp_lagged_col_df[col_label] = game_team_regular_season.groupby(['TEAM_ID', 'SEASON', 'HOME_VISITOR'])[col].transform(lambda x: x.shift(1).rolling(100, min_periods=1).agg(stat_type))

            

        game_team_regular_season = pd.concat([game_team_regular_season, temp_lagged_col_df], axis=1)


        del temp_lagged_col_df

        return game_team_regular_season
