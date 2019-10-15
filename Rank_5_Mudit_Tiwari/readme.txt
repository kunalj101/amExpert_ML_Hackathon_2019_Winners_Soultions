Running The Code:

There are three types of files:
1. Feature Engineering Files - Named as " Feature Engineering * "
2. Modeling Files            - Named as " *FILE ID* # *FILE NAME* " eg. 0-LGB-0 # Feat1_LGBM_Gruped_Cats_FE.ipynb
3. Ensembling File           - ensembling.ipynb


All these ipynb files will run in Jupyter Notebook. 

Please run the following files in the given order, also make sure that all the files given by AMEX is there in the current directory:

Sn       FILE NAME / FILE ID                             Running Time

1.  Feature Engineering 1.ipynb                            ~17 Mins 
2.  Feature Engineering 2.ipynb                            ~36 Mins
3.  Feature Engineering 3.ipynb                            ~30 Mins
4.  Feature Engineering 4.ipynb                            ~02 Mins
5.  Feature Engineering 5.ipynb                            ~02 Mins
6.  0-LGB-0 #                                              ~12 Mins 
7.  0-LGB-1 #                                              ~04 Mins
8.  0-LGB-2 #                                              ~04 Mins  
9.  0-CGB-0 #                                              ~30 Mins
10. 0-CGB-1 #                                              ~30 Mins
11. 0-XGB-0 #                                              ~30 Mins
12. 1-NN-0 # 											   ~05 Mins
13  1-LOG-0 #                                              ~05 Mins
14  ensembling.ipynb 									   ~01 Mins
                                                      ----------------
														   ~3 Hours

'OOT_ENSEMBLE_AVGD.csv' will be the output prediction file. 


ENVIRONMENT : Ubunutu 18.04 & Anaconda 1.7 (Latest)
			  16GB Memory (Recommended 32GB)* 
			 ~20GB HardDisk Space
		      8 Cores, i7 7th Gen.   
			  *A lot of RAM cleaning was required. 


Packages and Versioning:

# Name                    Version                   Build  Channel

anaconda                  2019.07                  py37_0
anaconda-client           1.7.2                    py37_0
anaconda-navigator        1.9.7                    py37_0
anaconda-project          0.8.3                      py_0
pandas                    0.24.2           py37he6710b0_0
numpy                     1.16.4           py37h7e9f1db_0
numpy-base                1.16.4           py37hde5b4d6_0
pandoc                    2.2.3.2                       0
pandocfilters             1.4.2                    py37_1
xgboost                   0.90                     pypi_0    pypi
lightgbm                  2.3.0                    pypi_0    pypi
catboost                  0.17.3                   pypi_0    pypi
keras                     2.3.0                    pypi_0    pypi
keras-applications        1.0.8                    pypi_0    pypi
keras-preprocessing       1.1.0                    pypi_0    pypi
tensorboard               2.0.0                    pypi_0    pypi
tensorflow                2.0.0                    pypi_0    pypi
tensorflow-estimator      2.0.0                    pypi_0    pypi
tqdm                      4.32.1                     py_0

