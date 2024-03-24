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



class TransformData():

    def __init__(self) -> None:

        self.static_cols = ['SEASON', 'TEAM_ID', 'TEAM_NAME', 'GAME_DATE_EST', 'HOME_VISITOR']

        self.cat_cols = ['HOME_VISITOR']

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
    







# TIME TREND: WE CREATE TWO TYPES: SHORT AND LONG FOR ANY FEATURES WE CREATE ---------------------------------------------

class CreateTimeBasedFeatures():

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

    def __init__(self) -> None:
        pass

    
