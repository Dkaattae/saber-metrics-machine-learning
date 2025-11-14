import os
import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def build_team_metrics(data):
    
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
    team_df[[f"{c}_prev" for c in cum_cols]] = (
            team_df.groupby('team')[cum_cols]
                    .cumsum()
                    .shift(fill_value=0)
    )

    team_df['OBP_prev'] = (
        (team_df['H_prev'] + team_df['BB_prev'] + team_df['HBP_prev']) / 
        (team_df['AB_prev'] + team_df['BB_prev'] + team_df['HBP_prev'] + team_df['SF_prev'])
    )
    team_df['SLG_prev'] = (
        (team_df['H_prev'] + 2*team_df['2B_prev'] + 
            3*team_df['3B_prev'] + 4*team_df['HR_prev']) / 
            team_df['AB_prev']
        ) 
    team_df['OPS_prev'] = team_df['OBP_prev'] + team_df['SLG_prev']

    team_df['FPCT_prev'] = (
        (team_df['putouts_prev'] + team_df['assists_prev']) / 
        (team_df['putouts_prev'] + team_df['assists_prev'] + team_df['errors_prev'])
    )
    output = team_df[["team", "game_number", "OPS_prev", "FPCT_prev"]].copy()
    output = output.rename(columns={
        "OPS_prev": "OPS",
        "FPCT_prev": "FPCT"
    })

    return team_df, output

def build_pitcher_metrics(data):
    home_pitcher_columns = ['home_P_id', 'home_game_number', 'game_length', 'away_HR', 'away_HBP', 'away_BB', 'away_SO', 'home_IP']
    away_pitcher_columns = ['away_P_id', 'away_game_number', 'game_length', 'home_HR', 'home_HBP', 'home_BB', 'home_SO', 'away_IP']

    pitch_columns = ['P_id', 'game_number', 'game_length', 'HR', 'HBP', 'BB', 'SO', 'IP']

    rename_homep = dict(zip(home_pitcher_columns, pitch_columns))
    rename_awayp = dict(zip(away_pitcher_columns, pitch_columns))

    homep_df = data[[c for c in home_pitcher_columns if c != 'home_IP']].copy()
    homep_df['home_IP'] = homep_df['game_length'].apply(lambda x: math.floor(x / 6))
    homep_df = homep_df.rename(columns=rename_homep)

    awayp_df = data[[c for c in away_pitcher_columns if c!= 'away_IP']].copy()
    awayp_df['away_IP'] = awayp_df['game_length'].apply(lambda x: math.ceil(x / 6))
    awayp_df = awayp_df.rename(columns=rename_awayp)

    pitcher_df = pd.concat([homep_df, awayp_df], ignore_index=True)

    pitcher_df = pitcher_df.sort_values(['P_id','game_number'])

    cum_cols = ['HR','HBP', 'BB', 'SO', 'IP']
    pitcher_df[[f"{c}_prev" for c in cum_cols]] = (
        pitcher_df.groupby('P_id')[cum_cols]
                .cumsum()
                .shift(fill_value=0)
    )

    fip_constant = 3.1
    pitcher_df['FIP_prev'] = ((13*pitcher_df['HR_prev'] + 
            3*(pitcher_df['BB_prev'] + pitcher_df['HBP_prev']) 
            - 2*pitcher_df['SO_prev']) / pitcher_df['IP_prev'].replace(0,np.nan)
             + fip_constant
    )
    output = pitcher_df[["P_id", "game_number", "FIP_prev"]].rename(
        columns={"FIP_prev": "FIP"}
    )

    return pitcher_df, output

def merge_all_metrics(data, team_metrics, pitcher_metrics, data_columns):
    df = data[data_columns].copy()
    df = df.merge(team_metrics.add_prefix('home_'), on=['home_team', 'home_game_number'], how='left')
    df = df.merge(team_metrics.add_prefix('away_'), on=['away_team', 'away_game_number'], how='left')
    df = df.merge(pitcher_metrics.add_prefix('home_'), on=['home_P_id', 'home_game_number'], how='left')
    df = df.merge(pitcher_metrics.add_prefix('away_'), on=['away_P_id', 'away_game_number'], how='left')

    df['home_won'] = (df['home_score']>df['away_score']).astype(int)
    return df

def transform_raw_data(year, raw_data_path, intermediate_path):
    data = pd.read_parquet(raw_data_path)
    data_columns = ['date', 'dayofweek', 'away_team', 'away_game_number', 'away_league', 'home_team', 'home_game_number', 'home_league', \
                    'home_score', 'away_score', 'park_id', 'away_P_id', 'home_P_id']
    team_df, team_metrics = build_team_metrics(data)
    pitcher_df, pitcher_metrics = build_pitcher_metrics(data)
    
    # team_df = team_df.rename(columns={c: c.replace('_prev','') for c in df.columns if '_prev' in c})
    # pitcher_df = pitcher_df.rename(columns={c: c.replace('_prev','') for c in df.columns if '_prev' in c})

    df_current = merge_all_metrics(data, team_metrics, pitcher_metrics, data_columns)
    
    df_current.to_parquet(intermediate_path)

    last_idx = team_metrics.groupby("team")["game_number"].idxmax()
    df_season = team_metrics.loc[last_idx].reset_index(drop=True)
    df_season['season'] = year

    last_idx_p = pitcher_metrics.groupby("P_id")["game_number"].idxmax()
    df_season_p = pitcher_metrics.loc[last_idx_p].reset_index(drop=True)
    df_season_p['season'] = year
    
    return (df_season, df_season_p)

if __name__ == "__main__":
    year_list = ['2021', '2022', '2023', '2024']
    df_season = pd.DataFrame()
    df_season_p = pd.DataFrame()
    if not os.path.exists('intermediate/'):
        os.mkdir('intermediate/')
    for year in year_list:
        raw_data_path = f'raw/data_{year}.parquet'
        intermediate_path = f'intermediate/cumsum_season_{year}.parquet'
        df_season_year, df_season_p_year = transform_raw_data(
            year, raw_data_path, intermediate_path)
        df_season = pd.concat([df_season, df_season_year], ignore_index=True)
        df_season_p = pd.concat([df_season_p, df_season_p_year], ignore_index=True)
    df_season.to_parquet('intermediate/team_season.parquet')
    df_season_p.to_parquet('intermediate/pitcher_season.parquet')

