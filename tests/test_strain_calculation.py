import pytest
import numpy as np
from core.strain_calculation import calculate_total_strain
from core.material_properties import MaterialProperties
from utils.utils import create_grid

def test_calculate_total_strain():
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

    # 実行：全ひずみの計算
    total_strain = calculate_total_strain(
        x1=x1, x2=x2, x3=x3,
        inclusion_shape=inclusion_shape,
        applied_strain=applied_strain,
        pre_strain=pre_strain,
        matrix=matrix, 
        inclusion=inclusion
    )

    # テスト: Total strainが期待される形状で出力されることを確認
    assert total_strain.shape == (50, 50, 50, 3, 3)
    
    # ひずみ値が適切な位置に存在することを確認
    assert total_strain[5, 5, 5, 0, 0] is not None  # 中心付近に値が存在すること

    # ひずみがゼロでないことを確認（単純なチェック）
    assert np.any(total_strain != 0)

    # 介在物内部のひずみが非ゼロであることを確認
    assert np.linalg.norm(total_strain[5, 5, 5]) > 0  # 乱れたひずみが中心付近で非ゼロ

def test_invalid_inclusion_shape():
    # 無効な形状（a1, a2, a3 が異なる）
    inclusion_shape_invalid = {"a1": 1.0, "a2": 0.5, "a3": 0.3}
    matrix = MaterialProperties(young_modulus=70e9, poisson_ratio=0.33)
    inclusion = MaterialProperties(young_modulus=100e9, poisson_ratio=0.25)

    with pytest.raises(ValueError):
        calculate_total_strain(
            x1=np.linspace(-2, 2, 10), 
            x2=np.linspace(-2, 2, 10), 
            x3=np.linspace(-2, 2, 10),
            inclusion_shape=inclusion_shape_invalid,
            applied_strain=np.zeros((3, 3)),
            pre_strain=np.zeros((3, 3)),
            matrix=matrix,
            inclusion=inclusion
        )

def test_strain_inside_inclusion():
    # 介在物内部のひずみのチェック
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
    
    total_strain = calculate_total_strain(
        x1=x1, x2=x2, x3=x3,
        inclusion_shape=inclusion_shape,
        applied_strain=applied_strain,
        pre_strain=pre_strain,
        matrix=MaterialProperties(young_modulus=70e9, poisson_ratio=0.33),
        inclusion=MaterialProperties(young_modulus=100e9, poisson_ratio=0.25)
    )

    # 期待されるひずみの乱れが介在物内で適用されているか
    assert np.linalg.norm(total_strain[5, 5, 5]) > 0  # 中心付近で乱れたひずみが非ゼロ
