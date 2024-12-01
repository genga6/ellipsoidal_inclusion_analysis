import numpy as np
from core.material_properties import MaterialProperties
from utils.utils import calculate_geometric_params


def _check_general_ellipsoid(a1, a2, a3):
    """
    Check the type of spheroid based on the axes lengths.

    :param a1: Length of the first axis (major or minor depending on shape)
    :param a2: Length of the second axis (minor or major depending on shape)
    :param a3: Length of the third axis (equal to a2 for oblate or prolate spheroid)
    :return: The type of spheroid ("sphere", "oblate", "prolate") or raises an error for invalid input
    """
    # Sphere: a1 == a2 == a3
    if a1 == a2 == a3:
        return "sphere"
    # oblate spheroid: a1 < a2 == a3
    elif a1 < a2 == a3:
        return "oblate"
    # prolate spheroid: a1 > a2 == a3
    elif a1 > a2 == a3:
        return "prolate"
    # Other cases are invalid for this function
    else:
        raise ValueError("For a spheroid, a1 must be either greater than or less than a2, with a2 equal to a3.")

def calculate_internal_eshelby_tensor(inclusion_shape, matrix):
    """
    Calculate the internal Eshelby tensor for a given inclusion shape and matrix properties.
    
    :param inclusion_shape: Dictionary containing the inclusion shape properties {"a1": float, "a2": float, "a3": float}.
    :param matrix_properties: Matrix material properties containing "shear_modulus", "poisson_ratio", etc.
    :return: 6x6 internal Eshelby tensor as a NumPy array.
    """
    a1, a2, a3 = inclusion_shape["a1"], inclusion_shape["a2"], inclusion_shape["a3"]
    poisson_ratio = matrix.poisson_ratio
    
    S = np.zeros((6, 6))  # Initialize Eshelby tensor
    
    # Check if the shape is spherical
    if a1 == a2 == a3:
        diag_value = (7 - 5 * poisson_ratio) / (15 * (1 - poisson_ratio))
        off_diag_value = (5 * poisson_ratio - 1) / (15 * (1 - poisson_ratio))
        shear_value = (4 - 5 * poisson_ratio) / (15 * (1 - poisson_ratio))

        # Diagonal and off-diagonal components
        for i in range(3):
            S[i, i] = diag_value
            for j in range(3):
                if i != j:
                    S[i, j] = off_diag_value

        # Shear components
        for i in range(3, 6):
            S[i, i] = shear_value
    
    # Check if the shape is prolate spheroid (a1 > a2 == a3)
    elif a1 > a2 == a3:
        I1, I2, I3, I11, I12, I13, I21, I22, I23, I31, I32, I33 = _calculate_integrals_prolate(a1, a2, a3)
        S = _assign_spheroid_tensor(a1, a2, a3, S, I1, I2, I3, I11, I12, I13, I21, I22, I23, I31, I32, I33, poisson_ratio)

    # Check if the shape is oblate spheroid (a1 < a2 == a3)
    elif a1 < a2 == a3:
        I1, I2, I3, I11, I12, I13, I21, I22, I23, I31, I32, I33 = _calculate_integrals_oblate(a1, a2, a3)
        S = _assign_spheroid_tensor(a1, a2, a3, S, I1, I2, I3, I11, I12, I13, I21, I22, I23, I31, I32, I33, poisson_ratio)

    else:
        raise ValueError("For a spheroid, a1 must be either greater than or less than a2, with a2 equal to a3.")

    return S


def _calculate_integrals_prolate(a1, a2, a3):
    """
    Calculate integrals I1, I2, I3 for prolate spheroids based on the given axes.
    
    :param a1: Length of the major axis (longer axis)
    :param a2: Length of the minor axis (shorter axis)
    :param a3: Length of the third axis (same as a2)
    :return: Tuple containing integrals (I1, I2, I3, I11, I12, I13, I21, I22, I23, I31, I32, I33)
    """
    # 積分式計算（プロレート楕円体の場合）
    I2 = ( (2 * np.pi * a1 * a3**2) / ((a1**2 - a3**2)**(3/2)) ) * ( (a1 / a3) * np.sqrt((a1**2 / a3**2) - 1) - np.arccosh(a1 / a3) )
    I3 = I2  # Prolate spheroid symmetry: I2 = I3
    I1 = 4 * np.pi - 2 * I2
    I12 = (I2 - I1) / (a1**2 - a3**2)
    I13 = I12
    I11 = ( (4 * np.pi) / (a1**2) - 2 * I12 ) / 3
    I23 = (np.pi / a3**2) - (I2 - I1) / (4 * (a1**2 - a3**2))
    I22 = ( (4 * np.pi / a3**2) - I23 - (I2 - I1) / (a1**2 - a3**2) ) / 3
    I33 = I22
    I23 = I22  # Symmetry in prolate spheroids
    I21 = I12
    I31 = I12
    I32 = I23

    return I1, I2, I3, I11, I12, I13, I21, I22, I23, I31, I32, I33

def _calculate_integrals_oblate(a1, a2, a3):
    """
    Calculate integrals I1, I2, I3 for oblate spheroids based on the given axes.
    
    :param a1: Length of the minor axis (shorter axis)
    :param a2: Length of the major axis (longer axis)
    :param a3: Length of the third axis (same as a2)
    :return: Tuple containing integrals (I1, I2, I3, I11, I12, I13, I21, I22, I23, I31, I32, I33)
    """
    
    I3 = ( (2*np.pi*(a3**2)*a1) / ((a3**2 - a1**2)**(3/2)) ) * ( np.arccos(a1/a3) - (a1/a3)*( 1 - ((a1**2)/(a3**2)) )**(1/2) )
    I2 = I3
    I1 = 4*np.pi - 2*I3
    I31 = (I3 - I1) / (a1**2 - a3**2)
    I21 = I31
    I11 = (1/3)*((4*np.pi)/(a1**2) - 2*I31)
    I32 = (np.pi/(a3**2)) - (1/4)*I31
    I33 = I32
    I22 = I32
    I13 = I31
    I23 = I32
    I12 = I31
    
    return I1, I2, I3, I11, I12, I13, I21, I22, I23, I31, I32, I33





def _assign_spheroid_tensor(a1, a2, a3, S, I1, I2, I3, I11, I12, I13, I21, I22, I23, I31, I32, I33, poisson_ratio):
    """
    Assigns values to the Eshelby tensor for prolate and oblate spheroids based on given integrals.
    """
    # Prolate case: a1 > a2 == a3
    if a1 > a2 and a2 == a3:
        # S11, S12, S13 for prolate case
        S[0, 0] = (3 * (a1**2) * I11) / (8 * np.pi * (1 - poisson_ratio)) + ((1 - 2 * poisson_ratio) * I1) / (8 * np.pi * (1 - poisson_ratio))
        S[0, 1] = (a3**2 * I12) / (8 * np.pi * (1 - poisson_ratio)) - ((1 - 2 * poisson_ratio) * I1) / (8 * np.pi * (1 - poisson_ratio))
        S[0, 2] = (a3**2 * I13) / (8 * np.pi * (1 - poisson_ratio)) - ((1 - 2 * poisson_ratio) * I1) / (8 * np.pi * (1 - poisson_ratio))

        # S21, S22, S23 for prolate case
        S[1, 0] = S[0, 1]
        S[1, 1] = (3 * (a3**2) * I22) / (8 * np.pi * (1 - poisson_ratio)) + ((1 - 2 * poisson_ratio) * I2) / (8 * np.pi * (1 - poisson_ratio))
        S[1, 2] = (a3**2 * I23) / (8 * np.pi * (1 - poisson_ratio)) - ((1 - 2 * poisson_ratio) * I2) / (8 * np.pi * (1 - poisson_ratio))

        # S31, S32, S33 for prolate case
        S[2, 0] = S[0, 2]
        S[2, 1] = S[1, 2]
        S[2, 2] = (3 * (a3**2) * I33) / (8 * np.pi * (1 - poisson_ratio)) + ((1 - 2 * poisson_ratio) * I3) / (8 * np.pi * (1 - poisson_ratio))

        # S44, S55, S66 for prolate case
        S[3, 3] = ( (((a3)**2 + (a3)**2) * I23) / (16 * np.pi * (1 - poisson_ratio)) ) + ( ((1 - 2 * poisson_ratio) * (I2 + I3)) / (16 * np.pi * (1 - poisson_ratio)) )
        S[4, 4] = ( (((a3)**2 + (a1)**2) * I31) / (16 * np.pi * (1 - poisson_ratio)) ) + ( ((1 - 2 * poisson_ratio) * (I3 + I1)) / (16 * np.pi * (1 - poisson_ratio)) )
        S[5, 5] = ( (((a1)**2 + (a3)**2) * I12) / (16 * np.pi * (1 - poisson_ratio)) ) + ( ((1 - 2 * poisson_ratio) * (I1 + I2)) / (16 * np.pi * (1 - poisson_ratio)) )


    # Oblate case: a1 < a2 == a3
    elif a1 < a2 and a2 == a3:
        # S11, S12, S13 for oblate case
        S[0, 0] = (3 * (a1**2) * I11) / (8 * np.pi * (1 - poisson_ratio)) + ((1 - 2 * poisson_ratio) * I1) / (8 * np.pi * (1 - poisson_ratio))
        S[0, 1] = (a3**2 * I12) / (8 * np.pi * (1 - poisson_ratio)) - ((1 - 2 * poisson_ratio) * I1) / (8 * np.pi * (1 - poisson_ratio))
        S[0, 2] = (a3**2 * I13) / (8 * np.pi * (1 - poisson_ratio)) - ((1 - 2 * poisson_ratio) * I1) / (8 * np.pi * (1 - poisson_ratio))

        # S21, S22, S23 for oblate case
        S[1, 0] = S[0, 1]
        S[1, 1] = (3 * (a3**2) * I22) / (8 * np.pi * (1 - poisson_ratio)) + ((1 - 2 * poisson_ratio) * I2) / (8 * np.pi * (1 - poisson_ratio))
        S[1, 2] = (a3**2 * I23) / (8 * np.pi * (1 - poisson_ratio)) + ((1 - 2 * poisson_ratio) * I2) / (8 * np.pi * (1 - poisson_ratio))

        # S31, S32, S33 for oblate case
        S[2, 0] = S[0, 2]
        S[2, 1] = S[1, 2]
        S[2, 2] = (3 * (a3**2) * I33) / (8 * np.pi * (1 - poisson_ratio)) + ((1 - 2 * poisson_ratio) * I3) / (8 * np.pi * (1 - poisson_ratio))

        # S44, S55, S66
        # S44, S55, S66
        S[3, 3] = ( (((a1)**2 + (a3)**2) * I23) / (16 * np.pi * (1 - poisson_ratio)) ) + ( ((1 - 2 * poisson_ratio) * (I2 + I3)) / (16 * np.pi * (1 - poisson_ratio)) )
        S[4, 4] = ( (((a3)**2 + (a1)**2) * I31) / (16 * np.pi * (1 - poisson_ratio)) ) + ( ((1 - 2 * poisson_ratio) * (I3 + I1)) / (16 * np.pi * (1 - poisson_ratio)) )
        S[5, 5] = ( (((a1)**2 + (a1)**2) * I12) / (16 * np.pi * (1 - poisson_ratio)) ) + ( ((1 - 2 * poisson_ratio) * (I1 + I2)) / (16 * np.pi * (1 - poisson_ratio)) )


    return S







def calculate_external_eshelby_tensor(x1, x2, x3, inclusion_shape: dict[str, float], matrix: MaterialProperties):
    """
    Calculate the external Eshelby tensor for a given inclusion shape and matrix properties.
    
    :param inclusion_shape: Dictionary containing the inclusion shape properties
                            {"a1": float, "a2": float, "a3": float}.
    :param matrix: Matrix material properties containing "shear_modulus", "poisson_ratio", etc.
    :return: 6x6 external Eshelby tensor as a NumPy array.
    """
    a1, a2, a3 = inclusion_shape["a1"], inclusion_shape["a2"], inclusion_shape["a3"]
    _check_general_ellipsoid(a1, a2, a3)

    poisson_ratio = matrix.poisson_ratio

    # Calculate rho, lambda, and other intermediate variables for the tensor calculation
    r, lambda_vals, rho, n1, n2, n3 = calculate_geometric_params(x1, x2, x3, a1, a2, a3)
    
    # Calculate G components (6x6 tensor) based on the input shape and properties for all grid points
    G = _calculate_G_matrix(x1, x2, x3, poisson_ratio, rho, n1, n2, n3)
    
    return G


def _calculate_G_matrix(x1, x2, x3, poisson_ratio, rho, n1, n2, n3):
    """
    Calculate the G matrix (6x6) for the exterior-point Eshelby tensor for all provided grid points.

    :param x1, x2, x3: Arrays representing the coordinates, typically from a meshgrid.
    :param poisson_ratio: Poisson's ratio of the material.
    :param rho: Scalar field of rho values obtained from geometric calculations.
    :param n1, n2, n3: Normal vector components obtained from geometric calculations.
    :return: A 6x6 tensor with additional dimensions representing the grid points.
    """
    # Pre-compute terms to improve efficiency
    rho_squared = rho**2
    common_factor = rho**3 / (30 * (1 - poisson_ratio))
    poisson_diff2 = 1 - 2 * poisson_ratio

    # Terms for G matrix calculation
    term_1 = 3 * rho_squared + 10 * poisson_ratio - 5
    term_2 = 3 * rho_squared - 10 * poisson_ratio + 5
    term_3 = 15 * (1 - rho_squared)
    term_4 = 15 * (poisson_diff2 - rho_squared)
    term_5 = 15 * (poisson_ratio - rho_squared)
    term_6 = 15 * (7 * rho_squared - 5)

    # Calculate squares and products of normal vector components
    n1_squared = n1**2
    n2_squared = n2**2
    n3_squared = n3**2
    n1_n2 = n1 * n2
    n1_n3 = n1 * n3
    n2_n3 = n2 * n3

    # Create an array to store G for all grid points
    G = np.zeros((6, 6, *x1.shape))

    # Diagonal components
    G[0, 0, ...] = common_factor * (term_1 + 2 * term_2 + term_3 * n1_squared + term_4 * n1_squared + term_5 * (4 * n1_squared) + term_6 * n1_squared**2)
    G[1, 1, ...] = common_factor * (term_1 + 2 * term_2 + term_3 * n2_squared + term_4 * n2_squared + term_5 * (4 * n2_squared) + term_6 * n2_squared**2)
    G[2, 2, ...] = common_factor * (term_1 + 2 * term_2 + term_3 * n3_squared + term_4 * n3_squared + term_5 * (4 * n3_squared) + term_6 * n3_squared**2)

    # Off-diagonal components (symmetry considerations)
    G[0, 1, ...] = common_factor * (term_3 * n1_n2 + term_4 * n1_n2 + term_6 * n1_squared * n2_squared + term_5 * n1 * n2)
    G[1, 0, ...] = G[0, 1, ...]

    G[0, 2, ...] = common_factor * (term_3 * n1_n3 + term_4 * n1_n3 + term_6 * n1_squared * n3_squared + term_5 * n1 * n3)
    G[2, 0, ...] = G[0, 2, ...]

    G[1, 2, ...] = common_factor * (term_3 * n2_n3 + term_4 * n2_n3 + term_6 * n2_squared * n3_squared + term_5 * n2 * n3)
    G[2, 1, ...] = G[1, 2, ...]

    # Shear components
    G[3, 3, ...] = common_factor * (term_3 * n2_n3 + term_6 * n1_squared * n2 * n3 + term_5 * n2 * n3)
    G[4, 4, ...] = common_factor * (term_3 * n3 * n1 + term_6 * n1_squared * n3_squared + term_5 * n3 * n1)
    G[5, 5, ...] = common_factor * (term_3 * n1_n2 + term_6 * n1_squared * n2_squared + term_5 * n1 * n2)

    G[3, 4, ...] = common_factor * (term_3 * n3 * n1 + term_6 * n1_squared * n3_squared + term_5 * n3 * n1)
    G[4, 3, ...] = G[3, 4, ...]

    G[3, 5, ...] = common_factor * (term_3 * n2 * n3 + term_6 * n1_squared * n2 * n3 + term_5 * n2 * n3)
    G[5, 3, ...] = G[3, 5, ...]

    G[4, 5, ...] = common_factor * (term_3 * n1_n2 + term_6 * n1_squared * n2_squared + term_5 * n1 * n2)
    G[5, 4, ...] = G[4, 5, ...]

    return G


if __name__ == "__main__":
    # Test code for the Eshelby tensor calculation
    matrix_properties = MaterialProperties(young_modulus=70e9, poisson_ratio=0.33)
    
    # Calculate internal Eshelby tensor
    inclusion_shape_prolate = {"a1": 1.0, "a2": 0.5, "a3": 0.5}  # Prolate spheroid
    S_internal = calculate_internal_eshelby_tensor(inclusion_shape_prolate, matrix_properties)
    print("Internal Eshelby Tensor（Prolate Case）")
    print(S_internal)

    # Calculate internal Eshelby tensor
    inclusion_shape_oblate = {"a1": 0.1, "a2": 1.0, "a3": 1.0}  # oblate spheroid
    S_internal = calculate_internal_eshelby_tensor(inclusion_shape_oblate, matrix_properties)
    print("Internal Eshelby Tensor（oblate Case）")
    print(S_internal)

    # Sphere case test
    inclusion_shape_sphere = {"a1": 1.0, "a2": 1.0, "a3": 1.0}  # Sphere
    S_internal_sphere = calculate_internal_eshelby_tensor(inclusion_shape_sphere, matrix_properties)
    print("\nInternal Eshelby Tensor (Sphere Case):")
    print(S_internal_sphere)




