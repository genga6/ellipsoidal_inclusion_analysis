import numpy as np
from core.material_properties import MaterialProperties
from core.disturbed_strain import calculate_disturbed_strain

def calculate_stress(
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
    Calculate the exterior stress based on the total strain and material properties.

    :param x1, x2, x3: Coordinate grids for the x, y, and z directions.
    :param inclusion_shape: Dictionary containing the inclusion shape properties: {"a1": float, "a2": float, "a3": float}.
    :param applied_strain: Applied strain as a 3x3 numpy array.
    :param pre_strain: Pre-strain within the inclusion as a 3x3 numpy array.
    :param matrix: Matrix material properties containing "young_modulus" and "poisson_ratio".
    :param inclusion: Inclusion material properties containing "young_modulus" and "poisson_ratio".
    :return: Exterior stress as a 3x3xNxNxN numpy array for each grid point.
    """
    # グリッドの次元を取得
    grid_shape = x1.shape

    # 外部全ひずみの計算
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

    # 弾性係数行列の計算
    C_m = np.zeros((6, 6))
    C_m[0, 0] = C_m[1, 1] = C_m[2, 2] = matrix.bulk_modulus + 4.0 / 3.0 * matrix.shear_modulus
    C_m[0, 1] = C_m[0, 2] = C_m[1, 0] = C_m[1, 2] = C_m[2, 0] = C_m[2, 1] = matrix.bulk_modulus - 2.0 / 3.0 * matrix.shear_modulus
    C_m[3, 3] = C_m[4, 4] = C_m[5, 5] = matrix.shear_modulus

    # 外部応力の計算
    stress_field = np.zeros_like(disturbed_strain_field)

    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                # 全ひずみの計算（すでに内部と外部の情報を含む）
                total_strain = disturbed_strain_field[i, j, k] + applied_strain

                # 主応力成分の計算
                stress_field[i, j, k, 0, 0] = (C_m[0, 0] * total_strain[0, 0] +
                                               C_m[0, 1] * total_strain[1, 1] +
                                               C_m[0, 2] * total_strain[2, 2])
                stress_field[i, j, k, 1, 1] = (C_m[1, 0] * total_strain[0, 0] +
                                               C_m[1, 1] * total_strain[1, 1] +
                                               C_m[1, 2] * total_strain[2, 2])
                stress_field[i, j, k, 2, 2] = (C_m[2, 0] * total_strain[0, 0] +
                                               C_m[2, 1] * total_strain[1, 1] +
                                               C_m[2, 2] * total_strain[2, 2])

                # せん断応力成分の計算
                stress_field[i, j, k, 1, 2] = stress_field[i, j, k, 2, 1] = C_m[3, 3] * (total_strain[1, 2] + total_strain[2, 1])
                stress_field[i, j, k, 0, 2] = stress_field[i, j, k, 2, 0] = C_m[4, 4] * (total_strain[0, 2] + total_strain[2, 0])
                stress_field[i, j, k, 0, 1] = stress_field[i, j, k, 1, 0] = C_m[5, 5] * (total_strain[0, 1] + total_strain[1, 0])

    return stress_field

# この関数は、ひずみ行列から外部応力を計算するPythonスクリプトです。
# MATLABのコードを参考にし、外部および内部応力を計算しています。