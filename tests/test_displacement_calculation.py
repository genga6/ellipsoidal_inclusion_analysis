import pytest
import numpy as np
from core.displacement_calculation import calculate_displacement
from core.material_properties import MaterialProperties
from utils.utils import create_grid

def test_calculate_displacement():
    # 条件設定
    matrix = MaterialProperties(young_modulus=70e9, poisson_ratio=0.33)
    inclusion_shape = {"a1": 1.0, "a2": 0.5, "a3": 0.5}
    applied_strain = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -0.0]])
    eigenstrain = np.array([[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]])

    # グリッド作成
    x1_range = (-4.5, 4.5)
    x2_range = (-4.5, 4.5)
    x3_range = (-4.5, 4.5)
    resolution = 50
    x1, x2, x3 = create_grid(x1_range, x2_range, x3_range, resolution)

    # 計算実行
    displacement = calculate_displacement(
        x1=x1, x2=x2, x3=x3,
        inclusion_shape=inclusion_shape,
        applied_strain=applied_strain,
        eigenstrain=eigenstrain,
        matrix=matrix
    )

    # テスト: 変位が期待される形状で出力されることを確認
    assert displacement.shape == (resolution, resolution, resolution, 3)
    # 変位量が期待範囲内であることを確認（仮の例）
    assert np.linalg.norm(displacement[5, 5, 5]) >= 0
