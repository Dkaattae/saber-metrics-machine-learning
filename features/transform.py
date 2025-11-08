import pandas as pd
import numpy as np
import math

import matplotlib.pyplot as plt

year = '2022'
def transform_raw_data(year):
    data = pd.read_parquet(f'../data/data_{year}.parquet')

    data_columns = ['date', 'dayofweek', 'away_team', 'away_game_number', 'away_league', 'home_team', 'home_game_number', 'home_league', \
                    'home_score', 'away_score', 'park_id', 'away_P_id', 'home_P_id']
    away_columns = ['away_team', 'away_game_number', 'away_AB', 'away_H', 'away_2B', 'away_3B', 'away_HR', 'away_SF', 'away_HBP', 'away_BB', 'away_SO', \
                'away_p_cnt', 'away_putouts', 'away_assists', 'away_errors']
    home_columns = ['home_team', 'home_game_number', 'home_AB', 'home_H', 'home_2B', 'home_3B', 'home_HR', \
                'home_SF', 'home_HBP', 'home_BB', 'home_SO', 'home_p_cnt', 'home_putouts', 'home_assists', 'home_errors']
    team_columns = ['team', 'game_number', 'AB', 'H', '2B', '3B', 'HR', 'SF', 'HBP', 'BB', 'SO', 'p_cnt', 'putouts', 'assists', 'errors']

    rename_home = dict(zip(home_columns, team_columns))
    rename_away = dict(zip(away_columns, team_columns))

    home_df = data[home_columns]
    home_df = home_df.rename(columns=rename_home)

    away_df = data[away_columns]
    away_df = away_df.rename(columns=rename_away)

    team_df = pd.concat([home_df, away_df], ignore_index=True)

    team_df = team_df.sort_values(['team','game_number'])

    cum_cols = ['AB','H','2B','3B','HR','SF','HBP','BB', 'putouts', 'assists', 'errors']
    team_df[cum_cols] = team_df.groupby('team')[cum_cols].cumsum().shift(fill_value=0)

    team_df['OBP'] = (team_df['H'] + team_df['BB'] + team_df['HBP']) / (team_df['AB'] + team_df['BB'] + team_df['HBP'] + team_df['SF'])
    team_df['SLG'] = (team_df['H'] + 2*team_df['2B'] + 3*team_df['3B'] + 4*team_df['HR']) / team_df['AB']
    team_df['OPS'] = team_df['OBP'] + team_df['SLG']

    team_df['FPCT'] = (team_df['putouts'] + team_df['assists']) / (team_df['putouts'] + team_df['assists'] + team_df['errors'])

    team_metrics_prev = team_df.copy()
    team_metrics_prev['game_number'] = team_metrics_prev.groupby('team')['game_number'].shift(-1)

    home_pitcher_columns = ['home_P_id', 'home_game_number', 'game_length', 'home_HR', 'home_HBP', 'home_BB', 'home_SO', 'home_IP']
    away_pitcher_columns = ['away_P_id', 'away_game_number', 'game_length', 'away_HR', 'away_HBP', 'away_BB', 'away_SO', 'away_IP']

    pitch_columns = ['P_id', 'game_number', 'game_length', 'HR', 'HBP', 'BB', 'SO', 'IP']

    rename_homep = dict(zip(home_pitcher_columns, pitch_columns))
    rename_awayp = dict(zip(away_pitcher_columns, pitch_columns))

    homep_df = data[[c for c in home_pitcher_columns if c != 'home_IP']].copy()
    homep_df['home_IP'] = homep_df['game_length'].apply(lambda x: math.floor(x / 6))
    homep_df = homep_df.rename(columns=rename_homep)

    awayp_df = data[[c for c in away_pitcher_columns if c!= 'away_IP']].copy()
    awayp_df['away_IP'] = homep_df['game_length'].apply(lambda x: math.ceil(x / 6))
    awayp_df = awayp_df.rename(columns=rename_awayp)

    pitcher_df = pd.concat([homep_df, awayp_df], ignore_index=True)

    pitcher_df = pitcher_df.sort_values(['P_id','game_number'])

    cum_cols = ['HR','HBP', 'BB', 'SO', 'IP']
    pitcher_df[cum_cols] = pitcher_df.groupby('P_id')[cum_cols].cumsum().shift(fill_value=0)

    fip_constant = 3.1
    pitcher_df['FIP'] = (13*pitcher_df['HR'] + 3*(pitcher_df['BB'] + pitcher_df['HBP']) - 2*pitcher_df['SO']) / pitcher_df['IP'] + fip_constant

    pitcher_metrics_prev = pitcher_df.copy()
    pitcher_metrics_prev['game_number'] = pitcher_metrics_prev.groupby('P_id')['game_number'].shift(-1)

    df_merged = data[data_columns].merge(team_metrics_prev.add_prefix('home_'), on=['home_team', 'home_game_number'], how='left')
    df_merged = df_merged.merge(team_metrics_prev.add_prefix('away_'), on=['away_team', 'away_game_number'], how='left')
    df_merged = df_merged.merge(pitcher_metrics_prev.add_prefix('home_'), on=['home_P_id', 'home_game_number'], how='left')
    df_merged = df_merged.merge(pitcher_metrics_prev.add_prefix('away_'), on=['away_P_id', 'away_game_number'], how='left')

    df_merged['home_won'] = (df_merged['home_score']>df_merged['away_score']).astype(int)
    df_current = df_merged[data_columns+['home_OPS', 'home_FIP', 'home_FPCT', 'away_OPS', 'away_FIP', 'away_FPCT', 'home_won']]

    df_current.to_parquet(f'cumsum_season_{year}')

    last_idx = team_df.groupby("team")["game_number"].idxmax()
    df_season = team_df.loc[last_idx].reset_index(drop=True)
    df_season['season'] = year

    last_idx_p = pitcher_df.groupby("P_id")["game_number"].idxmax()
    df_season_p = pitcher_df.loc[last_idx_p].reset_index(drop=True)
    df_season_p['season'] = year

return (df_season, df_season_p)

if __name__ == "__main__":
    year_list = ['2022', '2023' '2024']
    for year in year_list:
        transform_raw_data(year)
