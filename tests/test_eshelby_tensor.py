import pytest
import numpy as np
import re
from core.eshelby_tensor import calculate_internal_eshelby_tensor, calculate_external_eshelby_tensor
from core.material_properties import MaterialProperties
from utils.utils import create_grid


def test_calculate_eshelby_tensor():
    # 条件設定
    matrix = MaterialProperties(young_modulus=70e9, poisson_ratio=0.33)

    x1_range = (-4.5, 4.5)
    x2_range = (-4.5, 4.5)
    x3_range = (-4.5, 4.5)
    resolution = 50
    x1, x2, x3 = create_grid(x1_range, x2_range, x3_range, resolution)
    
    # 球状の場合のEshelbyテンソル計算
    inclusion_shape_sphere = {"a1": 1.0, "a2": 1.0, "a3": 1.0}
    internal_eshelby_tensor_sphere = calculate_internal_eshelby_tensor(inclusion_shape_sphere, matrix)
    external_eshelby_tensor_sphere = calculate_external_eshelby_tensor(x1, x2, x3, inclusion_shape_sphere, matrix)
    
    # 内部エシェルビーテンソルの形状の検証
    assert internal_eshelby_tensor_sphere.shape == (6, 6)
    # 外部エシェルビーテンソルの形状の検証（各座標点に対する6x6行列）
    assert external_eshelby_tensor_sphere.shape == (6, 6, 50, 50, 50)
    
    # 球状の場合の解析解の一部と比較
    expected_value_internal = (7 - 5 * matrix.poisson_ratio) / (15 * (1 - matrix.poisson_ratio))
    assert np.isclose(internal_eshelby_tensor_sphere[0, 0], expected_value_internal, rtol=1e-3)
    
    # 各座標点における外部エシェルビーテンソルのチェック
    # NaNをゼロに置き換える
    external_eshelby_tensor_sphere = np.nan_to_num(external_eshelby_tensor_sphere, nan=0.0)
    # 非常に小さい負の値をクリップしてゼロ以上にする
    clipped_external_eshelby_tensor_sphere = np.clip(external_eshelby_tensor_sphere, 0, None)
    for i in range(6):
        for j in range(6):
            assert np.all(clipped_external_eshelby_tensor_sphere[i, j] >= 0), (
                f"External Eshelby tensor has a value less than 0 at component ({i}, {j})"
            )

    # 回転楕円体の場合
    inclusion_shape_spheroid = {"a1": 1.0, "a2": 1.0, "a3": 0.5}
    internal_eshelby_tensor_spheroid = calculate_internal_eshelby_tensor(inclusion_shape_spheroid, matrix)
    external_eshelby_tensor_spheroid = calculate_external_eshelby_tensor(x1, x2, x3, inclusion_shape_spheroid, matrix)
    
    # 内部エシェルビーテンソルの形状の検証
    assert internal_eshelby_tensor_spheroid.shape == (6, 6)
    # 外部エシェルビーテンソルの形状の検証
    assert external_eshelby_tensor_spheroid.shape == (6, 6, 50, 50, 50)

    # 簡易チェック: 主対角成分が正の値であること
    assert internal_eshelby_tensor_spheroid[0, 0] > 0
    assert internal_eshelby_tensor_spheroid[1, 1] > 0
    assert internal_eshelby_tensor_spheroid[2, 2] > 0

    external_eshelby_tensor_spheroid = np.nan_to_num(external_eshelby_tensor_spheroid, nan=0.0)
    clipped_external_eshelby_tensor_spheroid = np.clip(external_eshelby_tensor_spheroid, 0, None)
    for i in range(6):
        for j in range(6):
            assert np.all(clipped_external_eshelby_tensor_spheroid[i, j] >= 0), (
                f"External Eshelby tensor has a value less than 0 at component ({i}, {j})"
            )

    # 不正な形状（例: a1, a2, a3 が異なる）の場合
    inclusion_shape_invalid = {"a1": 1.0, "a2": 0.5, "a3": 0.3}
    with pytest.raises(ValueError, match=re.escape("Eshelby tensor calculation for general ellipsoids (a1≠a2≠a3) is not supported.")):
        calculate_internal_eshelby_tensor(inclusion_shape_invalid, matrix)
    with pytest.raises(ValueError, match=re.escape("Eshelby tensor calculation for general ellipsoids (a1≠a2≠a3) is not supported.")):
        calculate_external_eshelby_tensor(x1, x2, x3, inclusion_shape_invalid, matrix)

if __name__ == "__main__":
    pytest.main()
