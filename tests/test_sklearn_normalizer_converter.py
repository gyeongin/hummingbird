"""
Tests sklearn Normalizer converter
"""
import unittest

import numpy as np
import torch
from skl2pytorch import convert_sklearn
from skl2pytorch.common.data_types import Float32TensorType
from sklearn.preprocessing import Normalizer


class TestSklearnNormalizerConverter(unittest.TestCase):
    def test_normalizer_converter(self):
        data = np.array([[1, 2, 3], [4, 3, 0], [0, 1, 4], [0, 5, 6]], dtype=np.float32)

        for norm in ["l1", "l2", "max"]:
            model = Normalizer(norm=norm)
            model.fit(data)

            pytorch_model = convert_sklearn(model, [("input", Float32TensorType([4, 3]))])
            self.assertTrue(pytorch_model is not None)
            self.assertTrue(np.allclose(model.transform(data), pytorch_model(torch.from_numpy(data)).data.numpy()))


if __name__ == "__main__":
    unittest.main()
