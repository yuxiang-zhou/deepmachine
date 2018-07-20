import numpy as np
import menpo.io as mio
import menpo
from scipy.interpolate import interp1d
import scipy as sp

from pathlib import Path
from scipy.io import loadmat
from menpo.image import Image
from menpo.shape import PointCloud
from menpo.transform import Translation, Scale

from scipy.sparse import csr_matrix

bodypose_matrix = csr_matrix((16, 16))

bodypose_matrix[0, 1] = 1
bodypose_matrix[1, 2] = 1
bodypose_matrix[2, 6] = 1
bodypose_matrix[6, 3] = 1
bodypose_matrix[3, 4] = 1
bodypose_matrix[4, 5] = 1
bodypose_matrix[6, 7] = 1
bodypose_matrix[7, 8] = 1
bodypose_matrix[8, 9] = 1
bodypose_matrix[10, 11] = 1
bodypose_matrix[11, 12] = 1
bodypose_matrix[12, 7] = 1
bodypose_matrix[7, 13] = 1
bodypose_matrix[13, 14] = 1
bodypose_matrix[14, 15] = 1

bodypose_matrix[1, 0] = 1
bodypose_matrix[2, 1] = 1
bodypose_matrix[6, 2] = 1
bodypose_matrix[3, 6] = 1
bodypose_matrix[4, 3] = 1
bodypose_matrix[5, 4] = 1
bodypose_matrix[7, 6] = 1
bodypose_matrix[8, 7] = 1
bodypose_matrix[9, 8] = 1
bodypose_matrix[11, 10] = 1
bodypose_matrix[12, 11] = 1
bodypose_matrix[7, 12] = 1
bodypose_matrix[13, 7] = 1
bodypose_matrix[14, 13] = 1
bodypose_matrix[15, 14] = 1


jaw_indices = np.arange(0, 17)
lbrow_indices = np.arange(17, 22)
rbrow_indices = np.arange(22, 27)
upper_nose_indices = np.arange(27, 31)
lower_nose_indices = np.arange(31, 36)
leye_indices = np.arange(36, 42)
reye_indices = np.arange(42, 48)
outer_mouth_indices = np.arange(48, 60)
inner_mouth_indices = np.arange(60, 68)

parts_68 = (jaw_indices, lbrow_indices, rbrow_indices, upper_nose_indices,
            lower_nose_indices, leye_indices, reye_indices,
            outer_mouth_indices, inner_mouth_indices)

