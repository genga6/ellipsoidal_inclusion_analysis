import numpy as np
from core.material_properties import MaterialProperties


def calculate_displacement(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    inclusion_shape: dict[str, float],
    applied_strain: np.ndarray,
    eigenstrain: np.ndarray,
    matrix: MaterialProperties
) -> np.ndarray:
    """
    Calculate displacement distribution in the material.

    :param x1: x1 coordinates (1D or 3D array depending on grid structure).
    :param x2: x2 coordinates (1D or 3D array depending on grid structure).
    :param x3: x3 coordinates (1D or 3D array depending on grid structure).
    :param inclusion_shape: Dictionary containing the inclusion shape properties:
                            {"a1": float, "a2": float, "a3": float}.
    :param applied_strain: Applied strain as a 3x3 numpy array.
    :param eigenstrain: Eigenstrain within the inclusion as a 3x3 numpy array.
    :param matrix: Dictionary containing matrix material properties:
                   {"shear_modulus": float, "bulk_modulus": float, "lambda_lame": float}.
    :return: Displacement distribution as an NxNxN numpy array for each grid point.
    """
    # メッシュを生成
    grid_shape = x1.shape

    # displacement_field の初期化
    displacement_field = np.zeros(grid_shape + (3,))

    # Inclusion shape parameters
    a1, a2, a3 = inclusion_shape["a1"], inclusion_shape["a2"], inclusion_shape["a3"]

    # Calculate Lame parameters
    shear_modulus = matrix.shear_modulus
    bulk_modulus = matrix.bulk_modulus
    lambda_lame = matrix.lambda_lame


    # Loop over grid points to calculate displacement
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            for k in range(grid_shape[2]):
                # Current grid point coordinates
                point = np.array([x1[i, j, k], x2[i, j, k], x3[i, j, k]])

                # Check if the point is inside the inclusion
                normalized_coords = (point / np.array([a1, a2, a3])) ** 2
                is_inside = np.sum(normalized_coords) <= 1

                if is_inside:
                    # Displacement inside the inclusion
                    displacement = eigenstrain @ point
                else:
                    # Displacement outside the inclusion
                    displacement = applied_strain @ point

                # Store displacement vector
                displacement_field[i, j, k, :] = displacement

    return displacement_field
