from enum import Enum, auto
import numpy as np

class Dtype(Enum):
    FLOAT32 = auto()
    FLOAT64 = auto()
    INT32 = auto()
    INT64 = auto()

# Converts python and numpy types into dtypes -> DTYPE
PYTHON_TYPE_TO_DTYPE = {
    int: Dtype.INT64,
    float: Dtype.FLOAT64,
    np.int32: Dtype.INT32,
    np.int64: Dtype.INT64,
    np.float32: Dtype.FLOAT32,
    np.float64: Dtype.FLOAT64,
}