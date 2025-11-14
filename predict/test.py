import requests

url = 'http://localhost:9696/predict'

new_data = {
        'date': 20250320,
        'dayofweek': 'Thu',
        'away_league': 'NL',
        'home_league': 'NL',
        'park_id': 'CHI12',
        'home_OPS_blend': 0.85,
        'home_FIP_blend': 3.9,
        'home_FPCT_blend': 0.976,
        'away_OPS_blend': 0.78,
        'away_FIP_blend': 4.9,
        'away_FPCT_blend': 0.985
    }

response = requests.post(url, json=new_data)
print(response)

predictions = response.json()

if predictions['predicted_win']:
    print('Home Team gonna win!')
elif predictions['predicted_lose']:
    print('Home Team might lose...')
else:
    print('You will see')