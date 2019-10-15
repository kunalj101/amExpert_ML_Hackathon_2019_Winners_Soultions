import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences

from mllib.params import FieldNames, FileNames
from mllib.transformers import ListAggregation, ListLen
from mllib.utils import load_pickle, read_csv, save_npy


def prepare_coupon_data(df, brand_type_map, category_map):
    brand_types = []
    categories = []
    brands = []
    item_ids = []
    for _, row in tqdm(df.iterrows()):
        btype = [brand_type_map[val] for val in row[FieldNames.item_brand_type]]
        brand_types.append(btype)
        cat = [category_map[val] for val in row[FieldNames.item_category]]
        categories.append(cat)
        brands.append(row[FieldNames.item_set])
        item_ids.append(row[FieldNames.item_brand])
    brand_types = pad_sequences(brand_types, maxlen=101, padding='post', truncating='post', value=0)
    categories = pad_sequences(categories, maxlen=101, padding='post', truncating='post', value=0)
    brands = pad_sequences(brands, maxlen=101, padding='post', truncating='post', value=0)
    item_ids = pad_sequences(item_ids, maxlen=101, padding='post', truncating='post', value=0)
    arr = np.dstack((item_ids, brands, brand_types, categories))
    return arr


def brand_cat_mapes():
    item_data = read_csv(FileNames.item)
    brand_type_map = {v: i for i, v in enumerate(item_data[FieldNames.item_brand_type].unique())}
    category_map = {v: i for i, v in enumerate(item_data[FieldNames.item_category].unique())}
    return brand_type_map, category_map


def get_customer_history(flag, n=1000):
    in_file = getattr(FileNames, '{}_v2'.format(flag))
    out_file = getattr(FileNames, '{}_customer_hist_nn_data'.format(flag))

    hist_flag = flag
    if flag == 'val':
        hist_flag = 'tr'
    if flag == 'test':
        hist_flag = 'train'

    hist_file = getattr(FileNames, 'cust_{}_artifact1'.format(hist_flag))

    df = load_pickle(in_file)
    hist = load_pickle(hist_file)
    agg = ListAggregation(date_col=FieldNames.campaign_start_date,
                          user_col=FieldNames.customer_id,
                          key_col=FieldNames.item_set,
                          hist_artifact=hist)
    arr = agg.transform(df)
    padded_arr = []
    for row in arr:
        parr = pad_sequences(row, maxlen=n, padding='pre', truncating='pre', value=0, dtype='int32')
        padded_arr.append(parr)
    padded_arr = np.concatenate(padded_arr)
    save_npy(out_file, padded_arr)


def get_save_coupon_vecs(flag, brty_map, cat_map):
    in_file = getattr(FileNames, '{}_v2'.format(flag))
    out_file = getattr(FileNames, '{}_coupon_nn_data'.format(flag))
    df = load_pickle(in_file)
    coupon_vectors = prepare_coupon_data(df, brty_map, cat_map)
    save_npy(out_file, coupon_vectors)


def main():
    brand_type_map, category_map = brand_cat_mapes()

    get_save_coupon_vecs('tr', brand_type_map, category_map)
    get_save_coupon_vecs('val', brand_type_map, category_map)
    get_save_coupon_vecs('train', brand_type_map, category_map)
    get_save_coupon_vecs('test', brand_type_map, category_map)

    get_customer_history('tr', n=512)
    get_customer_history('val', n=512)
    get_customer_history('train', n=512)
    get_customer_history('test', n=512)


if __name__ == '__main__':
    main()

