import requests

url = 'http://localhost:9696/predict'
# url = 'https://predict-cool-mountain-8274.fly.dev/predict'

new_data = {
    "date": 20250328,
    "dayofweek": "Fri",
    "away_league": "NL",
    "home_league": "NL",
    "park_id": "CHI12",
    "home_OPS_blend": 0.7960,
    "home_FIP_blend": 4.6507,
    "home_FPCT_blend": 0.9866,
    "away_OPS_blend": 0.8829,
    "away_FIP_blend": 3.0032,
    "away_FPCT_blend": 0.9847
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