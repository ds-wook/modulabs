for model in lightgbm xgboost catboost
do
    python src/train.py models=$model
    python src/predict.py models=$model
done