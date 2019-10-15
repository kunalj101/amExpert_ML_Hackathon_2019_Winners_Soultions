import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
import datetime as dt
import lightgbm as lgb
from tqdm import tqdm_notebook as tqdm
import random
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import LabelEncoder
import gc
import json


train = pd.read_csv("../Data/train.csv")
test = pd.read_csv("../Data/test.csv")
campaign_data = pd.read_csv("../Data/campaign_data.csv")
coupon_item_mapping = pd.read_csv("../Data/coupon_item_mapping.csv")
customer_demographics = pd.read_csv("../Data/customer_demographics.csv")
customer_transaction_data = pd.read_csv("../Data/customer_transaction_data.csv")
item_data = pd.read_csv("../Data/item_data.csv")
sample_submission = pd.read_csv("../Data/sample_submission.csv")

customer_transaction_data.date = pd.to_datetime(customer_transaction_data.date)
campaign_data.start_date = pd.to_datetime(campaign_data.start_date, format = "%d/%m/%y")
campaign_data.end_date = pd.to_datetime(campaign_data.end_date, format = "%d/%m/%y")

campaign_data['campaign_duration'] = (campaign_data['end_date'] - campaign_data['start_date']).dt.days
campaign_data['campaign_type'] = campaign_data['campaign_type'].map(lambda x : 1 if x == 'Y' else 0)
customer_demographics['marital_status'] = customer_demographics['marital_status'].astype(str).map(lambda x : 1 if x == 'Married' else 0 if x == 'Single' else -1)
customer_demographics['age_range'] = customer_demographics['age_range'].map(lambda x : int(x.replace("+", "").split("-")[0]))
customer_demographics['no_of_children'] = customer_demographics['no_of_children'].fillna(0).astype(str).map(lambda x : int(x.replace("+", "")))
customer_demographics['family_size'] = customer_demographics['family_size'].fillna(0).astype(str).map(lambda x : int(x.replace("+", "")))
customer_demographics['age_range_by_income_bracket'] = customer_demographics['age_range'] / customer_demographics['income_bracket']

train['campaign_unique_coupon'] = train[['campaign_id', 'coupon_id']].groupby('campaign_id').transform('nunique')
train['campaign_unique_customer'] = train[['campaign_id', 'customer_id']].groupby('campaign_id').transform('nunique')

test['campaign_unique_coupon'] = test[['campaign_id', 'coupon_id']].groupby('campaign_id').transform('nunique')
test['campaign_unique_customer'] = test[['campaign_id', 'customer_id']].groupby('campaign_id').transform('nunique')

customer_transaction_data = customer_transaction_data.merge(item_data, on = 'item_id', how = 'left').merge(
    coupon_item_mapping, on = 'item_id', how = 'left')


campaign_transaction_data = []
campaign_cust_coupon_data = []
for row in tqdm(campaign_data.itertuples()):
    df = customer_transaction_data[(customer_transaction_data.date < row.start_date)]
    df['selling_price_per_item'] = df.selling_price/df.quantity
    df['other_discount_per_item'] = df.other_discount/df.quantity
    df['coupon_discount_per_item'] = df.coupon_discount/df.quantity
    df['total_discount'] = -1 * (df['other_discount'] + df['coupon_discount'])
    df['price_before_discount'] = df['selling_price'] + df['total_discount']
    df['total_discount_perc'] = df['total_discount'] / df['price_before_discount']
    df['item_best_deal'] = df[['item_id', 'selling_price_per_item']].groupby('item_id').cummin()
    df['item_curr_deal_vs_best_deal'] = (df['selling_price_per_item'] - df['item_best_deal'])*df.quantity

    df1 = df.copy()
    df = df.drop_duplicates(subset = [c for c in df.columns if c != "coupon_id"])
    campaign_customer_features = df[['customer_id', 'date']].groupby('customer_id').count().reset_index().rename(
                                                            columns = {'date' : 'campaign_customer_transaction_count'})
    campaign_customer_features['campaign_id'] = row.campaign_id
    campaign_customer_features['campaign_total_spend'] = df[['customer_id', 'selling_price']].groupby(
                                                                            'customer_id').sum().reset_index()['selling_price']
    campaign_months = set([(row.start_date + dt.timedelta(days = i)).month for i in range(row.campaign_duration)])
    
    campaign_customer_features['campaign_total_saving'] = df[['customer_id', 'total_discount']].groupby(
                                                                            'customer_id').sum().reset_index()['total_discount']
    campaign_customer_features['campaign_coupon_saving'] = df[['customer_id', 'coupon_discount']].groupby(
                                                                        'customer_id').sum().reset_index()['coupon_discount']
    campaign_customer_features['campaign_other_saving'] = df[['customer_id', 'other_discount']].groupby(
                                                                            'customer_id').sum().reset_index()['other_discount']
    cust_campaign_month_coupon_saving = df[df.date.dt.month.isin(campaign_months)][['customer_id', 'coupon_discount']].groupby(
                                            'customer_id').sum()['coupon_discount'].to_dict()
    campaign_customer_features['cust_campaign_month_coupon_saving'] = campaign_customer_features.customer_id.map(cust_campaign_month_coupon_saving)
    days_since_last_coupon_redemption = df[df.coupon_discount < 0][['customer_id', 'date']].groupby('customer_id'
                                                                                                   ).max().date.to_dict()
    campaign_customer_features['campaign_days_since_last_coupon_redemption'] = campaign_customer_features.customer_id.map(
                                                                                            days_since_last_coupon_redemption)
    campaign_customer_features['campaign_days_since_last_coupon_redemption'] = campaign_customer_features[
                                    'campaign_days_since_last_coupon_redemption'].map(lambda x : (row.start_date - x).days)

    days_since_last_transaction = df[['customer_id', 'date']].groupby('customer_id').max().date.to_dict()
    campaign_customer_features['campaign_days_since_last_transaction'] = campaign_customer_features.customer_id.map(
                                                                                                    days_since_last_transaction)
    campaign_customer_features['campaign_days_since_last_transaction'] = campaign_customer_features[
                                            'campaign_days_since_last_transaction'].map(lambda x : (row.start_date - x).days)

    campaign_customer_features['campaign_total_saving_perc'] = campaign_customer_features['campaign_total_saving'
                                                                    ] / campaign_customer_features['campaign_total_spend']
    campaign_customer_features['campaign_coupon_saving_frac'] = campaign_customer_features['campaign_coupon_saving'
                                                                    ] / campaign_customer_features['campaign_total_saving']
    campaign_customer_features['campaign_other_saving_frac'] = campaign_customer_features['campaign_other_saving'
                                                                    ] / campaign_customer_features['campaign_total_saving']
    campaign_customer_features['campaign_unique_item_purchased'] = df[['customer_id', 'item_id']].drop_duplicates().groupby(
                                                                            'customer_id').count().reset_index()['item_id']
    campaign_customer_features['campaign_unique_coupon_redeemed'] = df[['customer_id', 'coupon_id']].drop_duplicates().groupby(
                                                                            'customer_id').count().reset_index()['coupon_id']
    campaign_customer_features['campaign_unique_brand_purchased'] = df[['customer_id', 'brand']].drop_duplicates().groupby(
                                                                            'customer_id').count().reset_index()['brand']
    campaign_customer_features['campaign_most_frequently_purchased_item'] = df[['customer_id', 'item_id', 'date']].groupby(
                                                            ['customer_id', 'item_id']).count().reset_index()[['customer_id', 
                                                            'date']].groupby('customer_id').max().reset_index()['date']
    campaign_customer_features['campaign_most_frequently_purchased_brand'] = df[['customer_id', 'brand', 'date']].groupby(
                                                            ['customer_id', 'brand']).count().reset_index()[['customer_id', 
                                                            'date']].groupby('customer_id').max().reset_index()['date']
    campaign_customer_features['campaign_most_frequently_purchased_coupon_id'] = df1[['customer_id', 'coupon_id', 'date']].groupby(
                                                            ['customer_id', 'coupon_id']).count().reset_index()[['customer_id', 
                                                            'date']].groupby('customer_id').max().reset_index()['date']
    campaign_customer_features['campaign_mean_discount_perc_availed'] = df[['customer_id','total_discount_perc']].groupby(
                                                            'customer_id').mean().reset_index()['total_discount_perc']
    campaign_customer_features['campaign_best_deal_availed_frac'] = df[['customer_id','item_curr_deal_vs_best_deal']].groupby(
                                                            'customer_id').mean().reset_index()['item_curr_deal_vs_best_deal']
    df = df.sort_values(by = ['customer_id', 'date'])
    df['prev_date'] = df.date.shift(1)
    df['prev_date'][df.customer_id != df.customer_id.shift(1)] = None
    df['days_from_last_transaction'] = (df['date'] - df.prev_date).dt.days
    campaign_customer_features['campaign_transaction_frequency'] = df[['customer_id','days_from_last_transaction']].groupby(
                                                            'customer_id').mean().reset_index()['days_from_last_transaction']
    customer_category_wise_spend = df[['customer_id', 'category', 'selling_price']].groupby([
                            'customer_id', 'category']).sum().reset_index().pivot(index = 'customer_id',
                                columns = 'category',  values = 'selling_price')
    customer_category_wise_spend.columns = ["perc_spend_" + c for c in customer_category_wise_spend.columns]
    campaign_customer_features = campaign_customer_features.merge(customer_category_wise_spend.reset_index(), 
                                        on = 'customer_id', how = 'left').fillna(0)
    
    for v in customer_category_wise_spend.columns:
        campaign_customer_features[v] = campaign_customer_features[v] / campaign_customer_features.campaign_total_spend

   
    customer_category_wise_saving = df[['customer_id', 'category', 'total_discount']].groupby([
                                'customer_id', 'category']).sum().reset_index().pivot(index = 'customer_id',
                                columns = 'category', values = 'total_discount')
    customer_category_wise_saving.columns = ["perc_saving_" + c for c in customer_category_wise_saving.columns]
    campaign_customer_features = campaign_customer_features.merge(customer_category_wise_saving.reset_index(),
                                    on = 'customer_id', how = 'left').fillna(0)
    for v in customer_category_wise_saving.columns:
        campaign_customer_features[v] = campaign_customer_features[v] / campaign_customer_features.campaign_total_saving
    
    
    customer_category_wise_coupon_saving = df[['customer_id', 'category', 'coupon_discount']].groupby([
                                'customer_id', 'category']).sum().reset_index().pivot(index = 'customer_id', 
                                columns = 'category', values = 'coupon_discount')
    customer_category_wise_coupon_saving.columns = ["perc_coupon_saving_" + c for c in 
                                                    customer_category_wise_coupon_saving.columns]
    campaign_customer_features = campaign_customer_features.merge(customer_category_wise_coupon_saving.reset_index(), 
                                                                  on = 'customer_id', how = 'left').fillna(0)
    
    customer_brand_type_wise_coupon_saving = df[['customer_id', 'brand_type', 'coupon_discount']].groupby([
                        'customer_id', 'brand_type']).sum().reset_index().pivot(index = 'customer_id', columns = 'brand_type', 
                                    values = 'coupon_discount')
    customer_brand_type_wise_coupon_saving.columns = ["perc_coupon_saving_" + c for c in 
                                                      customer_brand_type_wise_coupon_saving.columns]
    campaign_customer_features = campaign_customer_features.merge(customer_brand_type_wise_coupon_saving.reset_index(), 
                                                                  on = 'customer_id', how = 'left').fillna(0)
    for v in customer_brand_type_wise_coupon_saving.columns:
        campaign_customer_features[v] = campaign_customer_features[v] / campaign_customer_features.campaign_coupon_saving
    
    campaign_transaction_data.append(campaign_customer_features)
    
      
    campaign_cust_coupon_features = df1[['customer_id', 'coupon_id', 'date']].groupby(['customer_id', 'coupon_id']
                                ).count().reset_index().rename(columns = {'date' : 'campaign_cust_coupon_transaction_count'})
    campaign_cust_coupon_features['campaign_id'] = row.campaign_id
    campaign_cust_coupon_features['cust_coupon_id_coupon_saving'] = df1[['customer_id', 'coupon_id', 'coupon_discount']].groupby(
                                                    ['customer_id', 'coupon_id']).sum().reset_index()['coupon_discount']
    campaign_cust_coupon_features['cust_coupon_item_curr_deal_vs_best_deal'] = df1[['customer_id', 'coupon_id', 'item_curr_deal_vs_best_deal']].groupby(
                                                    ['customer_id', 'coupon_id']).sum().reset_index()['item_curr_deal_vs_best_deal']
    
    customer_category_wise_coupon_saving = df1[['customer_id', 'coupon_id', 'category', 'coupon_discount']].groupby([
                                'customer_id', 'coupon_id', 'category']).sum().reset_index().pivot_table(index = ['customer_id',
                                'coupon_id'], columns = 'category', values = 'coupon_discount')
    customer_category_wise_coupon_saving.columns = ["perc_coupon_id_saving_" + c for c in 
                                                    customer_category_wise_coupon_saving.columns]
    campaign_cust_coupon_features = campaign_cust_coupon_features.merge(customer_category_wise_coupon_saving.reset_index(), 
                                                                  on = ['customer_id', 'coupon_id'], how = 'left').fillna(0)
    for v in customer_category_wise_coupon_saving.columns:
        campaign_cust_coupon_features[v] = campaign_cust_coupon_features[v] / campaign_cust_coupon_features.cust_coupon_id_coupon_saving
    
    customer_brand_type_wise_coupon_saving = df1[['customer_id','coupon_id', 'brand_type', 'coupon_discount']].groupby([
                        'customer_id','coupon_id', 'brand_type']).sum().reset_index().pivot_table(index = ['customer_id', 'coupon_id'], 
                                columns = 'brand_type', values = 'coupon_discount')
    customer_brand_type_wise_coupon_saving.columns = ["perc_coupon_id_saving_" + c for c in 
                                                      customer_brand_type_wise_coupon_saving.columns]
     
    campaign_cust_coupon_features = campaign_cust_coupon_features.merge(customer_brand_type_wise_coupon_saving.reset_index(), 
                                                                  on = ['customer_id', 'coupon_id'], how = 'left').fillna(0)
    for v in customer_brand_type_wise_coupon_saving.columns:
        campaign_cust_coupon_features[v] = campaign_cust_coupon_features[v] / campaign_cust_coupon_features.cust_coupon_id_coupon_saving
    
    days_since_last_coupon_redemption = df1[df1.coupon_discount < 0][['customer_id', 'coupon_id', 'date']].groupby(['customer_id',
                                                                                    'coupon_id']).max().date.to_dict()
    def get_days_since_last_coupon_redemption(cust, coupon):
        try:
            return (row.start_date - days_since_last_coupon_redemption[(cust, coupon)]).days
        except:
            return None
    campaign_cust_coupon_features['campaign_days_since_last_coupon_redemption_curr'] = [get_days_since_last_coupon_redemption(
                        row1.customer_id, row1.coupon_id) for row1 in campaign_cust_coupon_features.itertuples()]
    coupon_id_unique_cust_count = df1[['customer_id', 'coupon_id']].drop_duplicates().coupon_id.value_counts().to_dict()
    campaign_cust_coupon_features['coupon_id_unique_cust_count'] = campaign_cust_coupon_features.coupon_id.map(coupon_id_unique_cust_count)
    coupon_id_total_discount = df[['coupon_id', 'coupon_discount']].groupby('coupon_id').sum().coupon_discount.to_dict()
    campaign_cust_coupon_features['coupon_id_total_discount'] = campaign_cust_coupon_features.coupon_id.map(coupon_id_total_discount)
    coupon_mean_item_curr_deal_vs_best_deal = df1[['coupon_id', 'item_curr_deal_vs_best_deal']].groupby('coupon_id').mean().item_curr_deal_vs_best_deal.to_dict()
    campaign_cust_coupon_features['coupon_mean_item_curr_deal_vs_best_deal'] = campaign_cust_coupon_features.coupon_id.map(coupon_mean_item_curr_deal_vs_best_deal)
    
    campaign_cust_coupon_data.append(campaign_cust_coupon_features)
campaign_transaction_data = pd.concat(campaign_transaction_data)
campaign_cust_coupon_data = pd.concat(campaign_cust_coupon_data)
    

def get_latent_features(df, u,v, t, key, dim = 16):
    """
    u = feature whose latent vector is to be calculate
    u = feature against whole the latent vector is calculated
    t = time var
    """
    df = df.sort_values(by = [u,v,t])
    df['prev_t'] = df[t].shift(1)
    df['inv_time_diff'] = (df[t] - df['prev_t']).map(lambda x : x.days + (x.seconds+1)/(3600*24)).map(lambda x : np.exp(-7/x))
    df['inv_time_diff'][(df[u] != df[u].shift(1)) | (df[v] != df[v].shift(1))] = np.exp(-7/60)
    
    R_df = df[[u,v,'inv_time_diff']].groupby([u,v]).sum().reset_index()
    R_df.columns = [u, v, 'tf']
    n = df[u].nunique()
    R_df['idf'] = R_df[[u, v]].groupby(v).transform('count')[u].map(lambda x : np.log(n/x))
    R_df['tf_idf'] = R_df['tf']*R_df['idf']
    R_df['tf_idf'] = R_df['tf_idf'] - R_df[[u, 'tf_idf']].groupby(u).transform('mean').tf_idf
    
    u_ids = sorted(df[u].unique())
    v_ids = sorted(df[v].unique())
    u_dict = {el : i for i, el in enumerate(u_ids)}
    v_dict = {el : i for i, el in enumerate(v_ids)}

    R_df['u_id'] = R_df[u].map(u_dict)
    R_df['v_id'] = R_df[v].map(v_dict)
    rows = R_df.u_id.values
    cols = R_df.v_id.values
    data = R_df.tf_idf.values
    R = coo_matrix((data, (rows, cols)), shape=(len(u_ids), len(v_ids)))
    U, sigma, Vt = svds(R, k = dim)

    user_latent_features = pd.DataFrame(U)
    user_latent_features.columns = ['{}_latent_feature_{}'.format(key,i) for i in range(U.shape[1])]
    user_latent_features[u] = user_latent_features.index.map(lambda x : u_ids[x])
    del U, R_df, R, sigma, Vt
    return user_latent_features.set_index(u)

customer_item_latent_features = get_latent_features(customer_transaction_data, 'customer_id', 'item_id', 'date', 'customer_item', 16)
customer_brand_latent_features = get_latent_features(customer_transaction_data, 'customer_id', 'brand', 'date', 'customer_brand', 16)
customer_coupon_latent_features = get_latent_features(customer_transaction_data, 'customer_id', 'coupon_id', 'date', 'customer_coupon', 16)
                                                  
item_customer_latent_features = get_latent_features(customer_transaction_data, 'item_id','customer_id', 'date', 'coupon_customer', 32)
brand_customer_latent_features = get_latent_features(customer_transaction_data, 'brand','customer_id', 'date', 'coupon_brand_customer', 16)
coupon_customer_latent_features1 = get_latent_features(customer_transaction_data, 'coupon_id','customer_id', 'date', 'coupon_customer1', 32)
coupon_item_latent_features = get_latent_features(customer_transaction_data, 'coupon_id','item_id', 'date', 'coupon_item', 16)
coupon_brand_latent_features = get_latent_features(customer_transaction_data, 'coupon_id','brand', 'date', 'coupon_brand', 16)


cust_coupon_sim_item = pd.DataFrame(np.matmul(customer_item_latent_features.values, coupon_item_latent_features.values.T))
cust_coupon_sim_item = cust_coupon_sim_item.unstack().reset_index()
cust_coupon_sim_item.columns = ['customer_id', 'coupon_id', 'cust_coupon_sim_item']

cust_coupon_sim_brand = pd.DataFrame(np.matmul(customer_brand_latent_features.values, coupon_brand_latent_features.values.T))
cust_coupon_sim_brand = cust_coupon_sim_brand.unstack().reset_index()
cust_coupon_sim_brand.columns = ['customer_id', 'coupon_id', 'cust_coupon_sim_brand']

coupon_customer_latent_features = coupon_item_mapping.merge(item_customer_latent_features.reset_index(), on = 'item_id', 
                                    how = 'left').fillna(0).drop('item_id', axis =1).groupby('coupon_id').mean().reset_index()
coupon_brand_customer_latent_features = coupon_item_mapping.merge(item_data, on = 'item_id')[['coupon_id', 'brand']].merge(brand_customer_latent_features.reset_index(), on = 'brand', 
                                    how = 'left').fillna(0).drop('brand', axis =1).groupby('coupon_id').mean().reset_index()

coupon_categories = coupon_item_mapping.merge(item_data, on = 'item_id', how = 'left').sort_values(by = 'coupon_id')[[
                            'coupon_id', 'item_id', 'category']].groupby(['coupon_id', 'category']).count().reset_index().pivot(
                            index = 'coupon_id', columns = 'category', values = 'item_id').fillna(0)
coupon_categories.columns = ["coupon_num_items_in_" + c for c in coupon_categories.columns]

coupon_brand_type = coupon_item_mapping.merge(item_data, on = 'item_id', how = 'left').sort_values(by = 'coupon_id')[[
                            'coupon_id', 'item_id', 'brand_type']].groupby(['coupon_id', 'brand_type']).count().reset_index().pivot(
                            index = 'coupon_id', columns = 'brand_type', values = 'item_id').fillna(0)
coupon_brand_type.columns = ["coupon_num_items_in_" + c for c in coupon_brand_type.columns]
coupon_categories = coupon_categories.join(coupon_brand_type)

train1 = train.merge(campaign_transaction_data, on = ['customer_id', 'campaign_id'], how = 'left').merge(campaign_data, on = 'campaign_id', 
            how = 'left').merge(customer_item_latent_features.reset_index(), on = 'customer_id', how = 'left').merge(
            customer_brand_latent_features.reset_index(), on = 'customer_id', how = 'left').merge(
            coupon_brand_customer_latent_features, on = 'coupon_id', how = 'left').merge(
            coupon_customer_latent_features, on = 'coupon_id', how = 'left').merge(customer_demographics, on = 'customer_id', 
            how = 'left').merge(coupon_categories.reset_index(), on = 'coupon_id', how = 'left').merge(
            campaign_cust_coupon_data, on = ['customer_id', 'coupon_id', 'campaign_id'], how = 'left').merge(
            customer_coupon_latent_features, on = 'customer_id', how = 'left').merge(coupon_customer_latent_features1,
            on = 'coupon_id', how = 'left').merge(coupon_item_latent_features,
            on = 'coupon_id', how = 'left').merge(coupon_brand_latent_features,
            on = 'coupon_id', how = 'left').merge(cust_coupon_sim_item, on = ['customer_id', 'coupon_id'], how = 'left').merge(
            cust_coupon_sim_brand, on = ['customer_id', 'coupon_id'], how = 'left')

test1 = test.merge(campaign_transaction_data, on = ['customer_id', 'campaign_id'], how = 'left').merge(campaign_data, on = 'campaign_id', 
            how = 'left').merge(customer_item_latent_features.reset_index(), on = 'customer_id', how = 'left').merge(
            customer_brand_latent_features.reset_index(), on = 'customer_id', how = 'left').merge(
            coupon_brand_customer_latent_features, on = 'coupon_id', how = 'left').merge(
            coupon_customer_latent_features, on = 'coupon_id', how = 'left').merge(customer_demographics, on = 'customer_id', 
            how = 'left').merge(coupon_categories.reset_index(), on = 'coupon_id', how = 'left').merge(
            campaign_cust_coupon_data, on = ['customer_id', 'coupon_id', 'campaign_id'], how = 'left').merge(
            customer_coupon_latent_features, on = 'customer_id', how = 'left').merge(coupon_customer_latent_features1,
            on = 'coupon_id', how = 'left').merge(coupon_item_latent_features,
            on = 'coupon_id', how = 'left').merge(coupon_brand_latent_features,
            on = 'coupon_id', how = 'left').merge(cust_coupon_sim_item, on = ['customer_id', 'coupon_id'], how = 'left').merge(
            cust_coupon_sim_brand, on = ['customer_id', 'coupon_id'], how = 'left')


indep_vars = ['coupon_id', 'customer_id',
    'campaign_unique_coupon', 'campaign_unique_customer',
              'campaign_customer_transaction_count', 'campaign_total_spend', 'campaign_total_saving',
              'campaign_coupon_saving', 'campaign_other_saving',
              'campaign_total_saving_perc', 'campaign_unique_item_purchased', 'campaign_most_frequently_purchased_item', 
              'campaign_unique_brand_purchased', 'campaign_most_frequently_purchased_brand',
              'campaign_mean_discount_perc_availed',
              'campaign_best_deal_availed_frac',
              'perc_spend_Alcohol', 
              'perc_spend_Bakery', 'perc_spend_Dairy, Juices & Snacks', 'perc_spend_Flowers & Plants', 'perc_spend_Fuel', 
              'perc_spend_Garden', 'perc_spend_Grocery', 'perc_spend_Meat', 'perc_spend_Miscellaneous', 
              'perc_spend_Natural Products', 'perc_spend_Packaged Meat', 'perc_spend_Pharmaceutical',
              'perc_spend_Prepared Food', 'perc_spend_Restauarant', 'perc_spend_Salads', 'perc_spend_Seafood',
              'perc_spend_Skin & Hair Care', 'perc_spend_Travel', 'perc_spend_Vegetables (cut)', 'perc_saving_Alcohol', 
              'perc_saving_Bakery', 'perc_saving_Dairy, Juices & Snacks', 'perc_saving_Flowers & Plants', 'perc_saving_Fuel',
              'perc_saving_Garden', 'perc_saving_Grocery', 'perc_saving_Meat', 'perc_saving_Miscellaneous', 
              'perc_saving_Natural Products', 'perc_saving_Packaged Meat', 'perc_saving_Pharmaceutical', 
              'perc_saving_Prepared Food', 'perc_saving_Restauarant', 'perc_saving_Salads', 'perc_saving_Seafood', 
              'perc_saving_Skin & Hair Care', 'perc_saving_Travel', 'perc_saving_Vegetables (cut)', 'campaign_type',
              'campaign_duration', 'customer_item_latent_feature_0', 
              'customer_item_latent_feature_1', 'customer_item_latent_feature_2', 'customer_item_latent_feature_3',
              'customer_item_latent_feature_4', 'customer_item_latent_feature_5', 'customer_item_latent_feature_6', 
              'customer_item_latent_feature_7', 'customer_item_latent_feature_8', 'customer_item_latent_feature_9', 
              'customer_item_latent_feature_10', 'customer_item_latent_feature_11', 'customer_item_latent_feature_12', 
              'customer_item_latent_feature_13', 'customer_item_latent_feature_14', 'customer_item_latent_feature_15', 
              'coupon_customer_latent_feature_0', 'coupon_customer_latent_feature_1', 'coupon_customer_latent_feature_2', 
              'coupon_customer_latent_feature_3', 'coupon_customer_latent_feature_4', 'coupon_customer_latent_feature_5', 
              'coupon_customer_latent_feature_6', 'coupon_customer_latent_feature_7', 'coupon_customer_latent_feature_8', 
              'coupon_customer_latent_feature_9', 'coupon_customer_latent_feature_10', 'coupon_customer_latent_feature_11', 
              'coupon_customer_latent_feature_12', 'coupon_customer_latent_feature_13', 'coupon_customer_latent_feature_14', 
              'coupon_customer_latent_feature_15', 'coupon_customer_latent_feature_16', 'coupon_customer_latent_feature_17', 
              'coupon_customer_latent_feature_18', 'coupon_customer_latent_feature_19', 'coupon_customer_latent_feature_20', 
              'coupon_customer_latent_feature_21', 'coupon_customer_latent_feature_22', 'coupon_customer_latent_feature_23', 
              'coupon_customer_latent_feature_24', 'coupon_customer_latent_feature_25', 'coupon_customer_latent_feature_26', 
              'coupon_customer_latent_feature_27', 'coupon_customer_latent_feature_28', 'coupon_customer_latent_feature_29', 
              'coupon_customer_latent_feature_30', 'coupon_customer_latent_feature_31', 'age_range', 'marital_status', 
              'rented', 'family_size', 'no_of_children', 'income_bracket',
             'customer_brand_latent_feature_0', 'customer_brand_latent_feature_1', 'customer_brand_latent_feature_2', 
              'customer_brand_latent_feature_3', 'customer_brand_latent_feature_4', 'customer_brand_latent_feature_5', 
              'customer_brand_latent_feature_6', 'customer_brand_latent_feature_7', 'customer_brand_latent_feature_8', 
              'customer_brand_latent_feature_9', 'customer_brand_latent_feature_10', 'customer_brand_latent_feature_11', 
              'customer_brand_latent_feature_12', 'customer_brand_latent_feature_13', 'customer_brand_latent_feature_14',
              'customer_brand_latent_feature_15', 'coupon_brand_customer_latent_feature_0', 
              'coupon_brand_customer_latent_feature_1', 'coupon_brand_customer_latent_feature_2',
              'coupon_brand_customer_latent_feature_3', 'coupon_brand_customer_latent_feature_4', 
              'coupon_brand_customer_latent_feature_5', 'coupon_brand_customer_latent_feature_6', 
              'coupon_brand_customer_latent_feature_7', 'coupon_brand_customer_latent_feature_8', 
              'coupon_brand_customer_latent_feature_9', 'coupon_brand_customer_latent_feature_10', 
              'coupon_brand_customer_latent_feature_11', 'coupon_brand_customer_latent_feature_12', 
              'coupon_brand_customer_latent_feature_13', 'coupon_brand_customer_latent_feature_14', 
              'coupon_brand_customer_latent_feature_15', 
             'coupon_num_items_in_Bakery', 'coupon_num_items_in_Dairy, Juices & Snacks', 'coupon_num_items_in_Flowers & Plants',
              'coupon_num_items_in_Garden', 'coupon_num_items_in_Grocery', 'coupon_num_items_in_Meat', 
              'coupon_num_items_in_Miscellaneous', 'coupon_num_items_in_Natural Products', 'coupon_num_items_in_Packaged Meat', 
              'coupon_num_items_in_Pharmaceutical', 'coupon_num_items_in_Prepared Food', 'coupon_num_items_in_Restauarant', 
              'coupon_num_items_in_Salads', 'coupon_num_items_in_Seafood', 'coupon_num_items_in_Skin & Hair Care', 
              'coupon_num_items_in_Travel', 'coupon_num_items_in_Vegetables (cut)', 
              'coupon_num_items_in_Established', 
              'coupon_num_items_in_Local',
              'perc_coupon_saving_Alcohol', 'perc_coupon_saving_Bakery', 'perc_coupon_saving_Dairy, Juices & Snacks', 
              'perc_coupon_saving_Flowers & Plants', 'perc_coupon_saving_Fuel', 'perc_coupon_saving_Garden', 
              'perc_coupon_saving_Grocery', 'perc_coupon_saving_Meat', 'perc_coupon_saving_Miscellaneous', 
              'perc_coupon_saving_Natural Products', 'perc_coupon_saving_Packaged Meat', 'perc_coupon_saving_Pharmaceutical',
              'perc_coupon_saving_Prepared Food', 'perc_coupon_saving_Restauarant', 'perc_coupon_saving_Salads', 
              'perc_coupon_saving_Seafood', 'perc_coupon_saving_Skin & Hair Care', 'perc_coupon_saving_Travel', 
              'perc_coupon_saving_Vegetables (cut)','perc_coupon_saving_Established', 'perc_coupon_saving_Local',
               'campaign_cust_coupon_transaction_count', 'cust_coupon_id_coupon_saving', 'perc_coupon_id_saving_Bakery',
              'perc_coupon_id_saving_Dairy, Juices & Snacks', 'perc_coupon_id_saving_Flowers & Plants', 
              'perc_coupon_id_saving_Garden', 'perc_coupon_id_saving_Grocery', 'perc_coupon_id_saving_Meat', 
              'perc_coupon_id_saving_Miscellaneous', 'perc_coupon_id_saving_Natural Products',
              'perc_coupon_id_saving_Packaged Meat', 'perc_coupon_id_saving_Pharmaceutical', 
              'perc_coupon_id_saving_Prepared Food', 'perc_coupon_id_saving_Restauarant', 'perc_coupon_id_saving_Salads', 
              'perc_coupon_id_saving_Seafood', 'perc_coupon_id_saving_Skin & Hair Care', 'perc_coupon_id_saving_Travel', 
              'perc_coupon_id_saving_Vegetables (cut)','customer_coupon_latent_feature_0', 'customer_coupon_latent_feature_1', 
              'customer_coupon_latent_feature_2', 'customer_coupon_latent_feature_3', 'customer_coupon_latent_feature_4', 
              'customer_coupon_latent_feature_5', 'customer_coupon_latent_feature_6', 'customer_coupon_latent_feature_7', 
              'customer_coupon_latent_feature_8', 'customer_coupon_latent_feature_9', 'customer_coupon_latent_feature_10',
              'customer_coupon_latent_feature_11', 'customer_coupon_latent_feature_12', 'customer_coupon_latent_feature_13', 
              'customer_coupon_latent_feature_14', 'customer_coupon_latent_feature_15', 'coupon_customer1_latent_feature_0', 
              'coupon_customer1_latent_feature_1', 'coupon_customer1_latent_feature_2', 'coupon_customer1_latent_feature_3', 
              'coupon_customer1_latent_feature_4', 'coupon_customer1_latent_feature_5', 'coupon_customer1_latent_feature_6', 
              'coupon_customer1_latent_feature_7', 'coupon_customer1_latent_feature_8', 'coupon_customer1_latent_feature_9', 
              'coupon_customer1_latent_feature_10', 'coupon_customer1_latent_feature_11', 'coupon_customer1_latent_feature_12', 
              'coupon_customer1_latent_feature_13', 'coupon_customer1_latent_feature_14', 'coupon_customer1_latent_feature_15', 
              'coupon_customer1_latent_feature_16', 'coupon_customer1_latent_feature_17', 'coupon_customer1_latent_feature_18', 
              'coupon_customer1_latent_feature_19', 'coupon_customer1_latent_feature_20', 'coupon_customer1_latent_feature_21', 
              'coupon_customer1_latent_feature_22', 'coupon_customer1_latent_feature_23', 'coupon_customer1_latent_feature_24', 
              'coupon_customer1_latent_feature_25', 'coupon_customer1_latent_feature_26', 'coupon_customer1_latent_feature_27', 
              'coupon_customer1_latent_feature_28', 'coupon_customer1_latent_feature_29', 'coupon_customer1_latent_feature_30',
              'coupon_customer1_latent_feature_31',
              'perc_coupon_id_saving_Established', 'perc_coupon_id_saving_Local', 'campaign_most_frequently_purchased_coupon_id',
              'campaign_days_since_last_coupon_redemption', 'campaign_days_since_last_coupon_redemption_curr',
              'campaign_days_since_last_transaction','campaign_coupon_saving_frac', 'campaign_other_saving_frac',
              'coupon_id_unique_cust_count', 'coupon_id_total_discount', 'campaign_transaction_frequency',
              'coupon_mean_item_curr_deal_vs_best_deal', 'cust_coupon_item_curr_deal_vs_best_deal',
              'coupon_item_latent_feature_0', 'coupon_item_latent_feature_1', 'coupon_item_latent_feature_2', 
              'coupon_item_latent_feature_3', 'coupon_item_latent_feature_4', 'coupon_item_latent_feature_5', 
              'coupon_item_latent_feature_6', 'coupon_item_latent_feature_7', 'coupon_item_latent_feature_8', 
              'coupon_item_latent_feature_9', 'coupon_item_latent_feature_10', 'coupon_item_latent_feature_11', 
              'coupon_item_latent_feature_12', 'coupon_item_latent_feature_13', 'coupon_item_latent_feature_14', 
              'coupon_item_latent_feature_15', 'coupon_brand_latent_feature_0', 'coupon_brand_latent_feature_1', 
              'coupon_brand_latent_feature_2', 'coupon_brand_latent_feature_3', 'coupon_brand_latent_feature_4', 
              'coupon_brand_latent_feature_5', 'coupon_brand_latent_feature_6', 'coupon_brand_latent_feature_7', 
              'coupon_brand_latent_feature_8', 'coupon_brand_latent_feature_9', 'coupon_brand_latent_feature_10',
              'coupon_brand_latent_feature_11', 'coupon_brand_latent_feature_12', 'coupon_brand_latent_feature_13', 
              'coupon_brand_latent_feature_14', 'coupon_brand_latent_feature_15','campaign_unique_coupon_redeemed',

              'age_range_by_income_bracket',
              'cust_campaign_month_coupon_saving', 
              'cust_coupon_sim_item', 'cust_coupon_sim_brand',
             ]
try:
    remove_vars = json.load(open("./remove_vars_best.json"))
except:
    remove_vars = []
indep_vars = list(set([v for v in indep_vars if v not in remove_vars] + ['age_range', 'marital_status', 'rented', 'family_size',
       'no_of_children', 'income_bracket', 'age_range_by_income_bracket']))


params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective':'binary',
    'metric': {'auc'},
    'num_leaves': 130,
    'learning_rate': 0.011,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'verbose': 1,
    'min_data_in_leaf' : 1,
    'max_bin' : 255,
    'lambda_l1' : 0.00002,
    'lambda_l2' : 0.00001,
    'min_gain_to_split' : 0.001
}

random.seed(2)
all_ids = set(train.id)
dev_ids = []
val_ids = []
for i in range(5):
    if i == 4:
        val_ids_ = all_ids
    else:
        val_ids_ = set(random.sample(list(all_ids), 15673))
    val_ids.append(val_ids_)
    dev_ids.append(set(train.id) - val_ids_)
    all_ids = all_ids - val_ids_
print([len(d) for d in dev_ids], [len(d) for d in val_ids])

models = []
val_scores = {}
for dev_ids_, val_ids_ in zip(dev_ids,val_ids) :
    dev = train1[train1.id.isin(dev_ids_)]
    val = train1[train1.id.isin(val_ids_)]
    lgb_dev = lgb.Dataset(dev[indep_vars], dev['redemption_status'])
    lgb_val = lgb.Dataset(val[indep_vars], val['redemption_status'])

    model = lgb.train(params, lgb_dev, num_boost_round = 3000, valid_sets = ( lgb_val), early_stopping_rounds = 500,
                     verbose_eval = 100)
    models.append(model)
    pred_valid = model.predict(val[indep_vars])
    print(np.mean(pred_valid), val.redemption_status.mean())
    val_scores.update({_id: p for _id, p in zip(val.id, pred_valid*val.redemption_status.mean() / pred_valid.mean() )})

for i, m in tqdm(enumerate(models)):
    test['pred_{}'.format(i)] = m.predict(test1[indep_vars])
    test['pred_{}'.format(i)] = test['pred_{}'.format(i)].rank(ascending = True, pct = True)
test['redemption_status'] = test[['pred_{}'.format(i) for i in range(5)]].mean(axis = 1)

test[['id', 'redemption_status']].to_csv("submission.csv", index = False)