import os
import pandas as pd

# tau is the number of games, when you equally treat previous season and current season.
# try 20-40 out of 162 games.
# FIP_mean_adj is the value that if the pitcher does not have a previous season. 
# FIP median is around 4, 75% is 4.5.

def blend_season_and_current(year, tau_team=20, tau_pitcher = 50, FIP_mean_adj=4.2):
    # data path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, 'final')
    folder = os.path.dirname(file_path)
    os.makedirs(folder, exist_ok=True)
    
    current_data_path=os.path.join(BASE_DIR, 'intermediate', f'cumsum_season_{year}.parquet')
    season_team_path=os.path.join(BASE_DIR, 'intermediate', 'team_season.parquet')
    season_pitcher_path=os.path.join(BASE_DIR, 'intermediate', 'pitcher_season.parquet')
    final_path=os.path.join(BASE_DIR, 'final', f'{year}_data.parquet')

    current_df = pd.read_parquet(current_data_path)
    season_df = pd.read_parquet(season_team_path)
    pitcher_df = pd.read_parquet(season_pitcher_path)
    current_df = current_df.fillna(0)

    prev_df = season_df[season_df['season'] == str(int(year) - 1)]
    prev_cols = ['team', 'OPS', 'FPCT', 'season']
    prev_df = prev_df[prev_cols]
    merged = current_df.merge(prev_df, left_on = 'home_team', \
            right_on="team", how="left")
    merged = merged.rename(columns={
        "team": "team_home_prev",
        "OPS": "OPS_home_prev",
        "FPCT": "FPCT_home_prev",
        "season": "season_team_away"
    })

    for col in ['OPS', 'FPCT']:
        merged[f'home_{col}_blend'] = (merged['home_game_number'] * merged[f'home_{col}'] + 
            tau_team * merged[f'{col}_home_prev']
            ) / (merged['home_game_number'] + tau_team)

    merged = merged.merge(prev_df, left_on = 'away_team', \
            right_on="team", how="left")
    merged = merged.rename(columns={
        "team": "team_away_prev",
        "OPS": "OPS_away_prev",
        "FPCT": "FPCT_away_prev",
        "season": "season_team_away"
    })
    for col in ['OPS', 'FPCT']:
        merged[f'away_{col}_blend'] = (merged['away_game_number'] * merged[f'away_{col}'] + 
            tau_team * merged[f'{col}_away_prev']
            ) / (merged['away_game_number'] + tau_team)


    prevp_df = pitcher_df[pitcher_df['season'] == str(int(year) - 1)]
    prev_pcols = ['P_id', 'FIP', 'season']
    prevp_df = prevp_df[prev_pcols]
    merged = merged.merge(prevp_df, left_on = 'home_P_id', \
            right_on="P_id", how="left")
    merged = merged.rename(columns={
        "P_id": "P_id_home_prev",
        "FIP": "FIP_home_prev",
        "season": "season_P_home"
    })
    merged['FIP_home_prev'] = merged['FIP_home_prev'].fillna(FIP_mean_adj)
    merged[f'home_FIP_blend'] = (merged['home_game_number'] * merged[f'home_FIP'] + 
            tau_pitcher * merged[f'FIP_home_prev']
            ) / (merged['home_game_number'] + tau_pitcher)

    merged = merged.merge(prevp_df, left_on = 'away_P_id', \
            right_on="P_id", how="left")
    merged = merged.rename(columns={
        "P_id": "P_id_away_prev",
        "FIP": "FIP_away_prev",
        "season": "season_P_away"
    })
    merged['FIP_away_prev'] = merged['FIP_away_prev'].fillna(FIP_mean_adj)
    merged[f'away_FIP_blend'] = (merged['away_game_number'] * merged[f'away_FIP'] + 
            tau_pitcher * merged[f'FIP_away_prev']
            ) / (merged['away_game_number'] + tau_pitcher)


    needed_cols = ['date', 'dayofweek', 'away_team', 'away_game_number', \
            'away_league', 'home_team', 'home_game_number', 'home_league', \
            'home_score', 'away_score', 'park_id', 'away_P_id', 'home_P_id', \
            'home_OPS_blend', 'home_FIP_blend', 'home_FPCT_blend', 'away_OPS_blend', \
            'away_FIP_blend', 'away_FPCT_blend', 'home_won']

    final_df = merged[needed_cols]
    final_df.to_parquet(final_path)

    return None

if __name__ == "__main__":
    year_list = ['2022', '2023', '2024']
    for year in year_list:
        blend_season_and_current(year)