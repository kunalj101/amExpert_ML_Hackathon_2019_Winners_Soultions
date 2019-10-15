import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

from mllib.params import FieldNames, FileNames
from mllib.utils import read_csv


if __name__ == '__main__':
    coupon_item = read_csv(FileNames.coupon_item)
    data = np.ones(len(coupon_item), )
    A_sparse = csr_matrix((data, (coupon_item[FieldNames.coupon_id].values, coupon_item[FieldNames.item_id].values)))
    nmf = NMF(16)
    coupon_vectors = nmf.fit_transform(A_sparse)
    print("Done fitting model.")
    item_vectors = nmf.components_.T
    name = 'nmf'
    np.save('data/coupon_vectors_{}.npy'.format(name), coupon_vectors)
    np.save('data/item_vectors_{}.npy'.format(name), item_vectors)
    print("Done saving nmf vectors.")
