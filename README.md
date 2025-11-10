# saber-metrics-machine-learning
machine learning project to predict MLB regular sasson game win rate using saber metrics 

## Description
Baseball has over a century of statistics, and traditional Sabermetrics metrics only capture part of the story. Machine learning lets us uncover which features truly drive outcomes, reveal nonlinear patterns, and provide data-driven insights for player evaluation and strategy.

This project applies machine learning models, including logistic regression, random forests and XGBoost, to historical and engineered features from baseball data. The pipeline covers data processing, model training, and feature importance analysis, with a FastAPI service for serving predictions.

## Dataset
retrosheet regular season game log
run download.py to download related raw data
    path: data/raw/*.parqueet
run transform.py to transform raw data into season aggregated data 
    path: data/intermediate/*.parquet
run season_blend.py to blend current season aggregated data with previous
season data.
    path: data/final/*.parquet

note: paths could be passed into functions. functions should be refactored later. 

## Features
categorical features: date, ......
numerical features (engineered): home_OPS_blend, away_OPS_blend, ...

### feature engineering: 
**Metrics**
*Offensive Metrics*
OPS = OBP + SLG
OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
SLG = (H + 2*2B + 3*3B + 4*HR) / AB
*Defensive Metrics*

*Pitching Metrics*

**Aggregated**
offensive and defensive metrics calculated as a team. 
pitching metrics aggregated to starting pitcher. 
    if in a game home team has 5 pitchers played, the metrics has all metrics calculated onto the starting pitcher for simplicity. 
    it would require play by play data to get data for each pitcher. and hard to predict pitcher appreance other than starting pitcher. 

**Blend**
the first few games, team has not played much yet, data are not sufficient to predict anything. so i blend in previous season data, with bayesian method.
blend_metrics = (game_number*metric_current + tau*metric_previous) 
            / (game_number + tau)
tau is a parameter to decide how many previous seasonal data we want to blend in. tau is set to the game number that we equally treat current season and previous season. 
there are 162 games for each team total in regular season. set tau to 20 - 40 is reasonable.

## Models
**Classification**
target is if home team won. that is home score > away score

*Logistic regression*
rmse: 0.50

*Randowm Forest (n_estimator=200, max_depth=5)*
rmse: 0.61

*XGBoost* 
rmse:

## Hyperopt

## Deployment