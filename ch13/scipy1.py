import numpy as np
from scipy import sparse
eye = np.eye(4)
print('Numpy 배열 : \n {}'.format(eye))

# 희소행열 : 대부분 값이 0인 행열
# 희소 행열 에서 메모리를 절약 하기 위해 사용..
sparse_matrix = sparse.csr_matrix(eye)
print('Numpy 배열 : \n {}'.format(sparse_matrix))

data = np.ones(4)
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print('COO 표현 : \n {}'.format(eye_coo))
