VM Specs while working out these steps
Ram - 60 GB
CPU - 15 cores
Min free hard disk space - 50GB

Environment Setup:
Install Anaconda Version - 4.7.10
Create Virtual Environment with Python- 3.7.3
and do pip install -r requirements.txt to install all the dependencies

Execution Steps:
Run the following notebooks in order)

1. final_variable_creation.ipynb
Run time - around 15 mins
script generates different features datasets, which is used for final model

2. model-lgb-xgb.ipynb
Run time - around 30 mins
Merges all the dataset and builds LightGBM, XGBoost with different validation strategy 

3. model-catb.ipynb
Run time - around 1 hours
Merges all the dataset and builds Catboost with different validation strategy 

4. ensembling.ipynb
Run time - 2 mins
Takes the mean of the all predictions created in the step 3 and 4 and saves it as final submission

final_submission.csv is the final output file