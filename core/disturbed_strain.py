import numpy as np
from core.material_properties import MaterialProperties
from core.eshelby_tensor import calculate_internal_eshelby_tensor, calculate_external_eshelby_tensor
from core.eigenstrain import calculate_eigenstrain
from utils.utils import strain_to_voigt, voigt_to_strain


def calculate_disturbed_strain(
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
    Calculate the disturbed strain using the internal and external Eshelby tensors.

    :param x1, x2, x3: Coordinate grids for the x, y, and z directions.
    :param inclusion_shape: Dictionary containing the inclusion shape properties: {"a1": float, "a2": float, "a3": float}.
    :param applied_strain: Applied strain as a 3x3 numpy array.
    :param pre_strain: Pre-strain within the inclusion as a 3x3 numpy array.
    :param matrix: Matrix material properties containing "shear_modulus", "bulk_modulus", "lambda_lame".
    :param inclusion: Inclusion material properties containing "shear_modulus", "bulk_modulus", "lambda_lame".
    :return: Disturbed strain as a 3x3xNxNxN numpy array for each grid point.
    """
    # グリッドの次元を取得
    grid_shape = x1.shape

    # 各グリッドポイントにおけるひずみの乱れを格納するためのフィールドを初期化
    # ひずみの乱れは各ポイントごとに6要素のVoigt形式で表現される
    disturbed_strain_field = np.zeros(grid_shape + (3, 3))

    # 内部エシェルビーテンソルの計算（6x6行列）
    internal_eshelby_tensor = calculate_internal_eshelby_tensor(inclusion_shape, matrix)

    # 外部エシェルビーテンソルの計算（各グリッドポイントごとに6x6行列）
    external_eshelby_tensor = calculate_external_eshelby_tensor(x1, x2, x3, inclusion_shape, matrix)

    # ひずみの計算をVoigt形式に変換
    eigenstrain = calculate_eigenstrain(inclusion_shape, applied_strain, pre_strain, matrix, inclusion)
    eigenstrain_voigt = strain_to_voigt(eigenstrain)

    # グリッド全体をループして、各ポイントでのひずみの乱れを計算
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                # 現在のグリッドポイントの座標
                point = np.array([x1[i, j, k], x2[i, j, k], x3[i, j, k]])

                # 介在物の内部かどうかを判定
                normalized_coords = (point / np.array([inclusion_shape["a1"], inclusion_shape["a2"], inclusion_shape["a3"]])) ** 2
                is_inside = np.sum(normalized_coords) <= 1

                # 使用するエシェルビーテンソルの選択
                if is_inside:
                    # 介在物内の場合、内部エシェルビーテンソルを使用
                    disturbed_strain_voigt = np.dot(internal_eshelby_tensor, eigenstrain_voigt)
                else:
                    # 介在物外の場合、外部エシェルビーテンソルを各座標に応じて使用
                    external_eshelby_tensor_at_point = external_eshelby_tensor[:, :, i, j, k]
                    disturbed_strain_voigt = np.dot(external_eshelby_tensor_at_point, eigenstrain_voigt)

                # Voigt形式から3x3のひずみ行列に変換し、フィールドに格納
                disturbed_strain_field[i, j, k] = voigt_to_strain(disturbed_strain_voigt)

    # 3x3の行列として返す
    return disturbed_strain_field