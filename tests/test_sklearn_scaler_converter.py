"""
Tests scikit scaler converter.
"""
import unittest
import numpy as np
import torch
from skl2pytorch import convert_sklearn
from skl2pytorch.common.data_types import Float32TensorType
from sklearn.preprocessing import RobustScaler, MaxAbsScaler, MinMaxScaler, StandardScaler


class TestSklearnScalerConverter(unittest.TestCase):
    def test_robust_scaler_floats(self):
        model = RobustScaler(with_centering=False)
        data = np.array([[0.0, 0.0, 3.0], [1.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 2.0]], dtype=np.float32)
        model.fit(data)

        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([4, 3]))])
        self.assertTrue(pytorch_model is not None)
        self.assertTrue(np.allclose(model.transform(data), pytorch_model(torch.from_numpy(data)).data.numpy()))

        model = RobustScaler(with_centering=True)
        model.fit(data)
        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([4, 3]))])
        self.assertTrue(pytorch_model is not None)
        self.assertTrue(np.allclose(model.transform(data), pytorch_model(torch.from_numpy(data)).data.numpy()))

    def test_max_abs_scaler_floats(self):
        model = MaxAbsScaler()
        data = np.array([[0.0, 0.0, 3.0], [1.0, -1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, -2.0]], dtype=np.float32)
        model.fit(data)

        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([4, 3]))])
        self.assertTrue(pytorch_model is not None)
        self.assertTrue(np.allclose(model.transform(data), pytorch_model(torch.from_numpy(data)).data.numpy()))

    def test_min_max_scaler_floats(self):
        model = MinMaxScaler()
        data = np.array([[0.0, 0.0, 3.0], [1.0, -1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, -2.0]], dtype=np.float32)
        model.fit(data)

        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([4, 3]))])
        self.assertTrue(pytorch_model is not None)
        self.assertTrue(np.allclose(model.transform(data), pytorch_model(torch.from_numpy(data)).data.numpy()))

    def test_standard_scaler_floats(self):
        model = StandardScaler()
        data = np.array([[0.0, 0.0, 3.0], [1.0, -1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, -2.0]], dtype=np.float32)

        model.fit(data)

        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([4, 3]))])
        self.assertTrue(pytorch_model is not None)
        self.assertTrue(np.allclose(model.transform(data), pytorch_model(torch.from_numpy(data)).data.numpy()))

        model = StandardScaler(with_std=False)
        model.fit(data)

        pytorch_model = convert_sklearn(model, [("input", Float32TensorType([4, 3]))])
        self.assertTrue(pytorch_model is not None)
        self.assertTrue(np.allclose(model.transform(data), pytorch_model(torch.from_numpy(data)).data.numpy()))


if __name__ == "__main__":
    unittest.main()
