import numpy as np
from core.eshelby_tensor import calculate_internal_eshelby_tensor
from core.material_properties import MaterialProperties
from utils.utils import strain_to_voigt, voigt_to_strain

def calculate_eigenstrain(
    inclusion_shape: dict[str, float],
    applied_strain: np.ndarray,
    pre_strain: np.ndarray,
    matrix: MaterialProperties,
    inclusion: MaterialProperties
) -> np.ndarray:
    # Calculate the Eshelby tensor for the inclusion
    eshelby_tensor = calculate_internal_eshelby_tensor(inclusion_shape, matrix)

    # Material properties (as in your existing function)
    shear_modulus_m = matrix.shear_modulus
    bulk_modulus_m = matrix.bulk_modulus
    lambda_lame_m = matrix.lambda_lame
    shear_modulus_i = inclusion.shear_modulus
    bulk_modulus_i = inclusion.bulk_modulus
    lambda_lame_i = inclusion.lambda_lame

    # Eigenstrain calculation (as before)
    applied_strain_trace = np.trace(applied_strain)
    pre_strain_trace = np.trace(pre_strain)
    applied_strain_dev = applied_strain - (1 / 3) * applied_strain_trace * np.eye(3)
    pre_strain_dev = pre_strain - (1 / 3) * pre_strain_trace * np.eye(3)

    eigenstrain_vol = (
        ((bulk_modulus_i - bulk_modulus_m) * applied_strain_trace - bulk_modulus_i * pre_strain_trace)
        / ((4 * shear_modulus_m + 3 * bulk_modulus_m) - (4 * shear_modulus_i + 3 * bulk_modulus_i))
    )

    eigenstrain_dev = (
        (15 * shear_modulus_i * (pre_strain_dev - applied_strain_dev))
        / (15 * (shear_modulus_m - shear_modulus_i) + 2 * (shear_modulus_i + shear_modulus_m))
    )

    eigenstrain = eigenstrain_vol * np.eye(3) + eigenstrain_dev

    # Convert eigenstrain to Voigt notation
    eigenstrain_voigt = strain_to_voigt(eigenstrain)

    # Use the Eshelby tensor to adjust the eigenstrain
    adjusted_eigenstrain_voigt = np.dot(eshelby_tensor, eigenstrain_voigt)

    # Convert the adjusted eigenstrain back to 3x3 tensor form
    adjusted_eigenstrain = voigt_to_strain(adjusted_eigenstrain_voigt)

    return adjusted_eigenstrain
