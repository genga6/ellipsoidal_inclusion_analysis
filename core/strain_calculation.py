import numpy as np
from core.material_properties import MaterialProperties
from core.disturbed_strain import calculate_disturbed_strain

def calculate_total_strain(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    inclusion_shape: dict[str, float],
    applied_strain: np.ndarray,
    pre_strain: np.ndarray,
    matrix: MaterialProperties,
    inclusion: MaterialProperties
) -> np.ndarray:
    """
    Calculate the total strain, which is the sum of the disturbed strain and applied strain.

    :param x1, x2, x3: Coordinate grids for the x, y, and z directions.
    :param inclusion_shape: Dictionary containing the inclusion shape properties: {"a1": float, "a2": float, "a3": float}.
    :param applied_strain: Applied strain as a 3x3 numpy array.
    :param pre_strain: Pre-strain within the inclusion as a 3x3 numpy array.
    :param matrix: Matrix material properties containing "shear_modulus", "bulk_modulus", "lambda_lame".
    :param inclusion: Inclusion material properties containing "shear_modulus", "bulk_modulus", "lambda_lame".
    :return: Total strain as a 3x3xNxNxN numpy array for each grid point.
    """
    # グリッドの次元を取得
    grid_shape = x1.shape

    # ひずみの乱れを計算
    disturbed_strain_field = calculate_disturbed_strain(
        x1=x1,
        x2=x2,
        x3=x3,
        inclusion_shape=inclusion_shape,
        applied_strain=applied_strain,
        pre_strain=pre_strain,
        matrix=matrix,
        inclusion=inclusion
    )

    # 全ひずみフィールドを初期化
    total_strain_field = np.zeros_like(disturbed_strain_field)

    # Inclusion shape parameters
    inclusion_radii = np.array([inclusion_shape["a1"], inclusion_shape["a2"], inclusion_shape["a3"]])

    # 各グリッドポイントごとに処理
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                # 現在の点の位置
                point = np.array([x1[i, j, k], x2[i, j, k], x3[i, j, k]])

                # Inclusion 内部かどうかの判定
                normalized_coords = (point / inclusion_radii) ** 2
                is_inside = np.sum(normalized_coords) <= 1

                if is_inside:
                    # Inclusion 内部の場合：適用ひずみ + 事前ひずみ + ひずみの乱れ
                    total_strain_field[i, j, k] = applied_strain + pre_strain + disturbed_strain_field[i, j, k]
                else:
                    # Inclusion 外部の場合：適用ひずみ + ひずみの乱れ
                    total_strain_field[i, j, k] = applied_strain + disturbed_strain_field[i, j, k]

    return total_strain_field
