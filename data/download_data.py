import os
import pandas as pd

for year in ["2021", "2022", "2023", "2024"]:
        url = f"https://www.retrosheet.org/gamelogs/gl{year}.zip"

        related_columns = [
                {'index': 0, 'name': 'date'},
                {'index': 2, 'name': 'dayofweek'},
                {'index': 3, 'name': 'away_team'},
                {'index': 4, 'name': 'away_league'},
                {'index': 5, 'name': 'away_game_number'},
                {'index': 6, 'name': 'home_team'},
                {'index': 7, 'name': 'home_league'},
                {'index': 8, 'name': 'home_game_number'},
                {'index': 9, 'name': 'away_score'},
                {'index': 10, 'name': 'home_score'},
                {'index': 11, 'name': 'game_length'},
                {'index': 16, 'name': 'park_id'},
                {'index': 21, 'name': 'away_AB'},
                {'index': 22, 'name': 'away_H'},
                {'index': 23, 'name': 'away_2B'},
                {'index': 24, 'name': 'away_3B'},
                {'index': 25, 'name': 'away_HR'},
                {'index': 28, 'name': 'away_SF'},
                {'index': 29, 'name': 'away_HBP'},
                {'index': 30, 'name': 'away_BB'},
                {'index': 32, 'name': 'away_SO'},
                {'index': 38, 'name': 'away_p_cnt'},
                {'index': 43, 'name': 'away_putouts'},
                {'index': 44, 'name': 'away_assists'},
                {'index': 45, 'name': 'away_errors'},
                {'index': 49, 'name': 'home_AB'},
                {'index': 50, 'name': 'home_H'},
                {'index': 51, 'name': 'home_2B'},
                {'index': 52, 'name': 'home_3B'},
                {'index': 53, 'name': 'home_HR'},
                {'index': 56, 'name': 'home_SF'},
                {'index': 57, 'name': 'home_HBP'},
                {'index': 58, 'name': 'home_BB'},
                {'index': 60, 'name': 'home_SO'},
                {'index': 66, 'name': 'home_p_cnt'},
                {'index': 71, 'name': 'home_putouts'},
                {'index': 72, 'name': 'home_assists'},
                {'index': 73, 'name': 'home_errors'},
                {'index': 101, 'name': 'away_P_id'},
                {'index': 103, 'name': 'home_P_id'}
        ]

        col_indices = [col['index'] for col in related_columns]
        col_names = [col['name'] for col in related_columns]

        df = pd.read_csv(url, header=None, usecols=col_indices)
        df.columns = col_names
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(BASE_DIR, 'raw')
        folder = os.path.dirname(file_path)
        os.makedirs(folder, exist_ok=True)
        df.to_parquet(os.path.join(file_path, f'data_{year}.parquet'))
