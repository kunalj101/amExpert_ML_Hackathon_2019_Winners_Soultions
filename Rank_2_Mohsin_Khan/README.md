# amexpert_19
AV competition codes

Approach:
A look at campign dates suggested train and test campigns were split in time. So, I decided to use last 3 campigns (sorted by campign start date) from tain data as validation data.

Other important thing was to observe, how the problem was set. For each campaign, there are certain coupuns. We would want to know given a given a set of cutomers, which coupons would they redeem.
The standard way to treat this would be to convert this into classification problem. This is done by pairing all customers in your set with all coupons. Suppose, there N customers and M coupons, this would lead to N*M rows. This would lead to large no. of rows even for smal N and M. So, to efficietntly solve tis problem, one needs to resample the rows (one way would be to sample rows in train data where user didn't redeem coupon, so that dataset becomes slightly more balanced). It can be observed that for each cutomer, average no. of coupons is around 16-20.

Given all this context, it was highly likely that customer interacts with aleast one coupon from the ones paired with him. So, first thing we need to do is get ranking of coupons for each customer. Durther, we can rank customers based on their propensity to redeem coupons (based on customers train data).

Now, the important question would be how do you rank coupons for a customer. Well, we have customer transactions, wherein we know which items generally customer reddems coupon on. If coupon paired with customer has more of those items, it is likely he will redeem that coupon.

So, lets rank `(customer, coupon)` pairs based on items common with his historically transacted items. Similary, also rank customers by their `coupon redemption mean` in train data.

```
train = pd.read_csv('train.csv')
customer_transact = pd.read_csv('customer_transaction_data.csv')

# historically interacted items by customer
customer_historical_items1 = customer_transact.groupby('customer_id')['item_id'].apply(set)

# items for customer applied coupon in past
customer_historical_items2 = customer_transact.loc[customer_transact.coupon_discount != 0].groupby('customer_id')['item_id'].apply(set)

# customer redemption propensity
customer_redeem_mean = train.groupby('customer_id')['redempton_status'].mean()

coupon_items = pd.read_csv('coupon_item_mappings.csv')
coupon_items = coupon_items.groupby('coupon_id')['item_id'].apply(set)

# Map data to test
test = pd.read_csv('test.csv')
test['customer_hist_items1'] = test['customer_id'].map(customer_historical_items1)
test['customer_hist_items2'] = test['customer_id'].map(customer_historical_items2)
test['coupon_items'] = test['coupon_id'].map(coupon_items)
test['customer_redeem_mean'] = test['customer_id'].map(customer_redeem_mean)

# Get common items
test['num_common_items1'] = test.apply(lambda x: len(x['coupon_items'] & x['customer_hist_items1']), axis=1)
test['num_common_items2'] = test.apply(lambda x: len(x['coupon_items'] & x['customer_hist_items2']), axis=1)

# Now just sort rows based on common items and customer_redeem_mean
test = test.sort_values(by=['num_common_items2', 'num_common_items1', 'customer_mean'])

n = len(test)
test['row_rank'] = list(range(n))/n

# Using above `row_rank` without any model scores 0.89+ on LB.
```

Reproducing results:
* Change following parameter in mllib/params.py
    * FileName.train and FileName.test values to train and test filenames
    * DATA --> path where all data resides
* Run `bash reproduce_stuff.sh` in terminal
* Finally run all notebook in order --> Model_v8.ipynb, Model_v9.ipynb, Model_v12.ipynb, Model_v13.ipynb, Model_v14, Model_v15
* Final notebook will write out `sub_en5.csv` in `DATA` folder

