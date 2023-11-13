
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
