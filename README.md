# saber-metrics-machine-learning
machine learning project to predict MLB regular sasson game win rate using saber metrics     

## Description
Baseball has over a century of statistics, and traditional Sabermetrics metrics only capture part of the story. Machine learning lets us uncover which features truly drive outcomes, reveal nonlinear patterns, and provide data-driven insights for player evaluation and strategy.   

This project applies machine learning models, including logistic regression, random forests and XGBoost, to historical and engineered features from baseball data. The pipeline covers data processing, model training, and feature importance analysis, with a FastAPI service for serving predictions.   

## Dataset
retrosheet regular season game log.  
run download.py to download related raw data.  
    path: data/raw/*.parqueet   
run transform.py to transform raw data into season aggregated data    
    path: data/intermediate/*.parquet.  
run season_blend.py to blend current season aggregated data with previous
season data.   
    path: data/final/*.parquet.  

note: paths could be passed into functions. functions should be refactored later.    

## Features
categorical features: date, dayofweek, away team, away team league, home team, home team league, park id, away starting pitcher id, home starting pitcher id.   
numerical features (engineered): home_OPS_blend, away_OPS_blend, home_FPCT_blend, away_FPCT_blend, home_FIP_blend, away_FIP_blend.   
all categorical features are game metadata, should be able to know before a game starts.    
all numerical features are combination of season to date metrics and previous season metrics.    
if no previous season, previous season metrics is 0.   
if no current season previous metrics, current season metrics is 0.   

note: there might be some issues in divided by 0 when calculating FPCT


### SaberMetrics: 
**Metrics**
*Offensive Metrics*.  
OPS = OBP + SLG.  
OBP = (H + BB + HBP) / (AB + BB + HBP + SF).  
SLG = (H + 2*2B + 3*3B + 4*HR) / AB.  
*Defensive Metrics*.  
FPCT = (Putouts + Assits) / (Putouts + Assits + Errors).  

*Pitching Metrics*.  
FIP = (13 * HR + 3 * (BB+HP) - 2*K) / IP + FIP constant.  
FIP constant is set to 3.1.  

**Aggregated**.  
offensive and defensive metrics calculated as a team.   
pitching metrics aggregated to starting pitcher.   
after calculated team/pitcher aggregated data, then join to the original game log to find out previous season to date aggregated metrics.   

note: if in a game home team has 5 pitchers played, the metrics has all metrics calculated onto the starting pitcher for simplicity.   
    it would require play by play data to get data for each pitcher. and hard to predict pitcher appreance other than starting pitcher.   
 
**Blend**.  
the first few games, team has not played much yet, data are not sufficient to predict anything. so i blend in previous season data, with bayesian method.   
blend_metrics = (game_number*metric_current + tau*metric_previous) 
            / (game_number + tau).  
tau is a parameter to decide how many previous seasonal data we want to blend in. tau is set to the game number that we equally treat current season and previous season.    
there are 162 games for each team total in regular season. set tau to 20 - 40 is reasonable.    

## Models
**Classification**.  
target is if home team won. that is home score > away score.  

*Decision Tree*.  
using DictVectorizer to one hot encoded categorical features  
auc: 0.557.  

*Randowm Forest*.  
n_estimator=200, max_depth=3,min_samples_leaf=7   
auc: 0.580.  

*XGBoost*    
auc: 0.577  

## Hyperopt

## Threshold
win threshold = 0.51.  
lose threshold = 0.49.  
F1 = 0.68.  
if predicted in the middle, print not sure.  

## Deployment
```
cd predict
docker build -t myfastapi .
docker run --rm -p 9696:9696 myfastapi
```
after spinning up the docker container,   
open another terminal, run    
`python test.py`.  

or copy   
example test data.  
```
{
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
```
to localhost:9696/docs.  
try it out
