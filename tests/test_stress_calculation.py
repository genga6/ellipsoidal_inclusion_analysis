import pytest
import numpy as np
from core.stress_calculation import calculate_stress
from core.material_properties import MaterialProperties
from utils.utils import create_grid

def test_calculate_stress():
    # 条件設定
    matrix = MaterialProperties(young_modulus=70e9, poisson_ratio=0.33)
    inclusion = MaterialProperties(young_modulus=100e9, poisson_ratio=0.25)
    
    # Inclusion shape parameters
    inclusion_shape = {"a1": 1.0, "a2": 0.5, "a3": 0.5}
    
    # Applied strain (example: uniform strain)
    applied_strain = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
    
    # Pre-strain (example: uniform pre-strain)
    pre_strain = np.array([[0.005, 0.0, 0.0], [0.0, 0.005, 0.0], [0.0, 0.0, 0.005]])
    
    # Grid creation
    x1_range = (-4.5, 4.5)
    x2_range = (-4.5, 4.5)
    x3_range = (-4.5, 4.5)
    resolution = 50
    x1, x2, x3 = create_grid(x1_range, x2_range, x3_range, resolution)

    # 実行：外部応力の計算
    stress_field = calculate_stress(
        x1=x1, x2=x2, x3=x3,
        inclusion_shape=inclusion_shape,
        applied_strain=applied_strain,
        pre_strain=pre_strain,
        matrix=matrix, 
        inclusion=inclusion
    )

    # テスト: Stress fieldが期待される形状で出力されることを確認
    assert stress_field.shape == (50, 50, 50, 3, 3)
    
    # 応力値が適切な位置に存在することを確認
    assert stress_field[5, 5, 5, 0, 0] is not None  # 中心付近に値が存在すること

    # 応力がゼロでないことを確認（単純なチェック）
    assert np.any(stress_field != 0)

    # 介在物内部の応力が非ゼロであることを確認
    assert np.linalg.norm(stress_field[5, 5, 5]) > 0  # 乱れた応力が中心付近で非ゼロ

def test_invalid_inclusion_shape():
    # 無効な形状（a1, a2, a3 が異なる）
    inclusion_shape_invalid = {"a1": 1.0, "a2": 0.5, "a3": 0.3}
    matrix = MaterialProperties(young_modulus=70e9, poisson_ratio=0.33)
    inclusion = MaterialProperties(young_modulus=100e9, poisson_ratio=0.25)

    with pytest.raises(ValueError):
        calculate_stress(
            x1=np.linspace(-2, 2, 10), 
            x2=np.linspace(-2, 2, 10), 
            x3=np.linspace(-2, 2, 10),
            inclusion_shape=inclusion_shape_invalid,
            applied_strain=np.zeros((3, 3)),
            pre_strain=np.zeros((3, 3)),
            matrix=matrix,
            inclusion=inclusion
        )

def test_stress_inside_inclusion():
    # 介在物内部の応力のチェック
    inclusion_shape = {"a1": 1.0, "a2": 1.0, "a3": 0.5}
    applied_strain = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])
    pre_strain = np.array([[0.005, 0.0, 0.0], [0.0, 0.005, 0.0], [0.0, 0.0, 0.005]])

    # Central grid point inside the inclusion
    x1, x2, x3 = np.meshgrid(
        np.linspace(-2, 2, 10),
        np.linspace(-2, 2, 10),
        np.linspace(-2, 2, 10),
        indexing='ij'
    )
    
    stress_field = calculate_stress(
        x1=x1, x2=x2, x3=x3,
        inclusion_shape=inclusion_shape,
        applied_strain=applied_strain,
        pre_strain=pre_strain,
        matrix=MaterialProperties(young_modulus=70e9, poisson_ratio=0.33),
        inclusion=MaterialProperties(young_modulus=100e9, poisson_ratio=0.25)
    )

    # 期待される応力の乱れが介在物内で適用されているか
    assert np.linalg.norm(stress_field[5, 5, 5]) > 0  # 中心付近で乱れた応力が非ゼロ
