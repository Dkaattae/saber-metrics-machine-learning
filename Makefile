all: save_model

download:
	python data/download_data.py

transform: download
	python data/transform.py

blend: transform
	python data/season_blend.py

preprocess: blend
	python train/preprocess.py

hyperopt: preprocess
	python train/hpo.py
