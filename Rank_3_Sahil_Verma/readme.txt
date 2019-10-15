All the required packages are in requirements.txt file
I have already put the all the competition files in this folder. Please make sure the file structure is the same.

The code is divided into two folders, one for feature creation and other for modelling.
First we run the feature creation scripts. Order is as follows:
1) feature_creation_code/data_prep_v2.ipynb
2) feature_creation_code/fc_ideas.ipynb
3) feature_creation_code/feats_user_coup_item.ipynb
4) feature_creation_code/feats_user_coup_item_brand_feats.ipynb

This is will create 12 csvs in the current folder. 

Now we will run the modelling script. Order is as follows:
1) modelling/modelling_adding_camp_cust_more_feats.ipynb
2) modelling/modelling_adding_camp_cust_more_feats-xgb.ipynb
3) modelling/modelling_adding_camp_cust_more_feats-CATBOOST.ipynb
4) modelling/modelling_adding_camp_cust_more_feats_v2.ipynb
5) modelling/modelling_adding_camp_cust_more_feats_v2_xgb.ipynb
6) modelling/modelling_adding_camp_cust_more_feats_v2-CATBOOST.ipynb
7) modelling/modelling_adding_camp_cust_more_feats_group_k_camp_xgb_lgb.ipynb
8) modelling/modelling_adding_camp_cust_more_feats_group_k_camp_xgb_lgb_v2.ipynb
9) modelling/modelling_adding_camp_cust_more_feats_group_k_camp_CATBOOST.ipynb
10) modelling/modelling_adding_camp_cust_more_feats_group_k_camp_v2-CATBOOST.ipynb
11) modelling/ensembling.ipynb

final_sub_2.csv will contain the final submission

