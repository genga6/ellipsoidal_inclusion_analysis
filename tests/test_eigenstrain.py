import pytest
import numpy as np
from core.eigenstrain import calculate_eigenstrain
from core.material_properties import MaterialProperties

def test_calculate_eigenstrain():
    # 条件設定
    matrix = MaterialProperties(young_modulus=70e9, poisson_ratio=0.33)
    inclusion = MaterialProperties(young_modulus=210e9, poisson_ratio=0.33)
    
    # テストケース 1: a1 == a2 != a3 (Oblate spheroid)
    inclusion_shape = {"a1": 1.0, "a2": 0.5, "a3": 0.5}
    applied_strain = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -0.0]])
    pre_strain = np.array([[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])

    eigenstrain = calculate_eigenstrain(
        inclusion_shape=inclusion_shape,
        applied_strain=applied_strain,
        pre_strain=pre_strain,
        matrix=matrix,
        inclusion=inclusion
    )
    
    # Eigenstrainのサイズを確認
    assert eigenstrain.shape == (3, 3)
    assert np.allclose(eigenstrain, eigenstrain, atol=1e-3)

    # テストケース 2: a1 ≠ a2 ≠ a3 (General ellipsoid) - エラーケース
    inclusion_shape_invalid = {"a1": 1.0, "a2": 0.5, "a3": 0.3}
    with pytest.raises(ValueError):  # エラーメッセージをチェックするために match は省略
        calculate_eigenstrain(
            inclusion_shape=inclusion_shape_invalid,
            applied_strain=applied_strain,
            pre_strain=pre_strain,
            matrix=matrix,
            inclusion=inclusion
        )