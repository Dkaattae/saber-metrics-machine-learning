from transform import transform_raw_data

year = 2022
df_season, df_season_p = transform_raw_data(year)

# tau is the number of games, when you equally treat previous season and current season.
# try 20-40 out of 162 games.
# metric_blend = (game_number * metrics_current + tau * previous) / (game_number + tau)

# transform.py will have in season aggregated data and seasonal data returned. 
# blend in season aggregated and last season data. 