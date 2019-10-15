# Change train and test filepath under `FIleName` and `DATA` value in mllib/params.py
python -m amlib.prepare_train_test_data
python -m amlib.prepare_transaction_data
python -m amlib.prepare_artifacts
python -m amlib.generate_features
python -m amlib.generate_features_v2
python -m amlib.generate_featuresi_v3
