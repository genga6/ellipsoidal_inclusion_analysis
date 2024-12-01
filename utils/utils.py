import numpy as np

def create_grid(x1_range, x2_range, x3_range, resolution):
    """
    Create a 3D grid of coordinates for spatial analysis.
    
    :param x1_range: Tuple (min, max) for x1
    :param x2_range: Tuple (min, max) for x2
    :param x3_range: Tuple (min, max) for x3
    :param resolution: Number of points in each direction
    :return: Meshgrid for x1, x2, x3
    """
    x1 = np.linspace(x1_range[0], x1_range[1], resolution)
    x2 = np.linspace(x2_range[0], x2_range[1], resolution)
    x3 = np.linspace(x3_range[0], x3_range[1], resolution)
    return np.meshgrid(x1, x2, x3)

def strain_to_voigt(strain: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 strain tensor to a 6x1 Voigt notation vector.
    
    :param strain: 3x3 strain tensor.
    :return: 6x1 Voigt notation vector.
    """
    return np.array([
        strain[0, 0],  # ε_xx
        strain[1, 1],  # ε_yy
        strain[2, 2],  # ε_zz
        2 * strain[1, 2],  # γ_yz
        2 * strain[0, 2],  # γ_xz
        2 * strain[0, 1]   # γ_xy
    ])

def voigt_to_strain(voigt: np.ndarray) -> np.ndarray:
    """
    Convert a 6x1 Voigt notation vector to a 3x3 strain tensor.
    
    :param voigt: 6x1 Voigt notation vector.
    :return: 3x3 strain tensor.
    """
    return np.array([
        [voigt[0], voigt[5] / 2, voigt[4] / 2],
        [voigt[5] / 2, voigt[1], voigt[3] / 2],
        [voigt[4] / 2, voigt[3] / 2, voigt[2]]
    ])

def calculate_geometric_params(x1, x2, x3, a1, a2, a3):
    """
    Calculate geometric parameters (rho, lambda, and normal vectors) needed for the Eshelby tensor.
    
    :param a1, a2, a3: Semi-principal axes of the ellipsoid or spheroid.
    :return: Geometric parameters (r, lambda_vals, rho, n1, n2, n3).
    """
    # Calculate distance from the origin (r)
    r = np.sqrt(x1**2 + x2**2 + x3**2)

    # Calculate lambda
    lambda_vals = (r**2 - a1**2 - a2**2 + np.sqrt((r**2 + a1**2 - a2**2)**2 - 4 * (a1**2 - a2**2) * x1**2)) / 2
    # Ensure lambda_vals are non-negative to prevent sqrt from failing
    lambda_vals = np.maximum(lambda_vals, 0)

    # Calculate rho
    rho1 = a1 / np.sqrt(a1**2 + lambda_vals)
    rho2 = a2 / np.sqrt(a2**2 + lambda_vals)
    rho3 = a3 / np.sqrt(a3**2 + lambda_vals)
    rho = (rho1 * rho2 * rho3)**(1/3)

    # Calculate theta
    theta1 = x1 / (a1**2 + lambda_vals)
    theta2 = x2 / (a2**2 + lambda_vals)
    theta3 = x3 / (a3**2 + lambda_vals)
    theta = theta1**2 + theta2**2 + theta3**2

    # Calculate normal vector components
    n1 = x1 / ((a1**2 + lambda_vals) * np.sqrt(theta))
    n2 = x2 / ((a2**2 + lambda_vals) * np.sqrt(theta))
    n3 = x3 / ((a3**2 + lambda_vals) * np.sqrt(theta))
    
    return r, lambda_vals, rho, n1, n2, n3