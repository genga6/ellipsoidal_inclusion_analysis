import pytest
import numpy as np
from core.material_properties import MaterialProperties
from core.disturbed_strain import calculate_disturbed_strain, calculate_internal_eshelby_tensor, calculate_external_eshelby_tensor
from utils.utils import create_grid, strain_to_voigt, voigt_to_strain

def test_calculate_disturbed_strain():
    # 条件設定
    matrix = MaterialProperties(young_modulus=70e9, poisson_ratio=0.33)
    inclusion = MaterialProperties(young_modulus=100e9, poisson_ratio=0.25)

    # Inclusion shape parameters
    inclusion_shape = {"a1": 1.0, "a2": 1.0, "a3": 0.5}

    # Applied strain (example: uniform strain)
    applied_strain = np.array([[0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]])

    # Pre-strain (example: uniform pre-strain)
    pre_strain = np.array([[0.005, 0.0, 0.0], [0.0, 0.005, 0.0], [0.0, 0.0, 0.005]])

    # グリッド作成
    x1_range = (-4.5, 4.5)
    x2_range = (-4.5, 4.5)
    x3_range = (-4.5, 4.5)
    resolution = 50
    x1, x2, x3 = create_grid(x1_range, x2_range, x3_range, resolution)

    # 実行：ひずみの乱れを計算
    disturbed_strain = calculate_disturbed_strain(
        x1=x1, x2=x2, x3=x3,
        inclusion_shape=inclusion_shape,
        applied_strain=applied_strain,
        pre_strain=pre_strain,
        matrix=matrix,
        inclusion=inclusion
    )

    # テスト: 計算結果が期待される形状であることを確認
    assert disturbed_strain.shape == (resolution, resolution, resolution, 3, 3)  # 3x3テンソルが各格子点に格納されている

    # テスト: 乱れたひずみの計算結果が正しく反映されていること
    # 一部の格子点でのチェック（ここでは中心付近）
    center_idx = resolution // 2
    assert np.linalg.norm(disturbed_strain[center_idx, center_idx, center_idx]) > 0  # 中心付近で乱れたひずみが非ゼロ

    # テスト: 内部と外部のエシェルビーテンソルの影響が正しく反映されているか
    # 中心付近のポイントを使用してエシェルビーテンソルによる期待値を計算
    point = np.array([x1[center_idx, center_idx, center_idx],
                      x2[center_idx, center_idx, center_idx],
                      x3[center_idx, center_idx, center_idx]])
    normalized_coords = (point / np.array([inclusion_shape["a1"], inclusion_shape["a2"], inclusion_shape["a3"]])) ** 2
    is_inside = np.sum(normalized_coords) <= 1

    # エシェルビーテンソルの計算
    if is_inside:
        internal_eshelby_tensor = calculate_internal_eshelby_tensor(inclusion_shape, matrix)
        expected_disturbed_strain_voigt = np.dot(internal_eshelby_tensor, strain_to_voigt(applied_strain + pre_strain))
    else:
        external_eshelby_tensor = calculate_external_eshelby_tensor(x1, x2, x3, inclusion_shape, matrix)
        external_eshelby_tensor_at_point = external_eshelby_tensor[:, :, center_idx, center_idx, center_idx]
        expected_disturbed_strain_voigt = np.dot(external_eshelby_tensor_at_point, strain_to_voigt(applied_strain + pre_strain))

    # Voigt形式から3x3のひずみ行列に変換
    expected_disturbed_strain = voigt_to_strain(expected_disturbed_strain_voigt)

    # 中心付近の値が期待される範囲内であることを確認
    assert np.allclose(disturbed_strain[center_idx, center_idx, center_idx], expected_disturbed_strain, rtol=1e-1)

def test_invalid_inclusion_shape():
    # 無効な形状（a1, a2, a3 が異なる）
    inclusion_shape_invalid = {"a1": 1.0, "a2": 0.5, "a3": 0.3}
    matrix = MaterialProperties(young_modulus=70e9, poisson_ratio=0.33)
    inclusion = MaterialProperties(young_modulus=210e9, poisson_ratio=0.33)

    # 一般的な楕円体に対してエラーが発生することを確認
    with pytest.raises(ValueError, match="Eshelby tensor calculation for general ellipsoids"):
        calculate_disturbed_strain(
            x1=np.linspace(-2, 2, 10), 
            x2=np.linspace(-2, 2, 10), 
            x3=np.linspace(-2, 2, 10),
            inclusion_shape=inclusion_shape_invalid,
            applied_strain=np.zeros((3, 3)),
            pre_strain=np.zeros((3, 3)),
            matrix=matrix,
            inclusion=inclusion
        )
