import numpy as np
import os
from core.material_properties import MaterialProperties
from core.eigenstrain import calculate_eigenstrain
from core.stress_calculation import calculate_stress
from core.strain_calculation import calculate_total_strain
from core.displacement_calculation import calculate_displacement
from core.eshelby_tensor import calculate_internal_eshelby_tensor
from utils.visualization_2d import plot_2d_results, extract_2d_slice
from utils.utils import create_grid


def main():
    """
    Main function to execute the ellipsoidal inclusion analysis.
    """
    # Define material properties for matrix and inclusion
    matrix = MaterialProperties(young_modulus=70e9, poisson_ratio=0.33)
    inclusion = MaterialProperties(young_modulus=210e9, poisson_ratio=0.33)

    # Define inclusion shape parameters (example: ellipsoid)
    inclusion_shape = {
        "a1": 1.0,  # Semi-axis along x1
        "a2": 1.0,  # Semi-axis along x2
        "a3": 1.0,  # Semi-axis along x3
    }

    # Define applied strain and pre-strain
    applied_strain = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0], 
    ])
    pre_strain = np.array([
        [-0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5], 
    ])
    
    # Define the grid and field generation (for example, creating a 3D meshgrid for x1, x2, x3)
    x1_range = (-4.5, 4.5)
    x2_range = (-4.5, 4.5)
    x3_range = (-4.5, 4.5)
    
    # Create a grid for the coordinates
    x1, x2, x3 = create_grid(x1_range, x2_range, x3_range, resolution=50)

        # Calculate Eshelby tensor
    eshelby_tensor = calculate_internal_eshelby_tensor(
        inclusion_shape=inclusion_shape,
        matrix=matrix
    )
    print("Eshelby Tensor:\n", eshelby_tensor)

    # Calculate eigenstrain
    eigenstrain = calculate_eigenstrain(
        inclusion_shape=inclusion_shape,
        applied_strain=applied_strain,
        pre_strain=pre_strain,
        matrix=matrix,
        inclusion=inclusion
    )
    print("Eigenstrain:\n", eigenstrain)

    # Example: Calculate the stress field based on the applied strain (or other computations)
    stress = calculate_stress(
        x1=x1, x2=x2, x3=x3,
        inclusion_shape=inclusion_shape,
        applied_strain=applied_strain,
        pre_strain=pre_strain,
        matrix=matrix, 
        inclusion=inclusion
    )
    
    # Choose a plane to slice (e.g., "x3", "x2", "x1")
    plane = "x2"
    value = 0.0  # Value along the chosen plane
    
    # Extract 2D slice of the stress field
    x2_slice, y2_slice, stress_slice = extract_2d_slice(x1, x2, x3, stress, plane=plane, value=value)
    print(x2_slice.shape, y2_slice.shape, stress_slice.shape)   # (50, 50) (50, 50) (50, 50, 3, 3)
    
    # Ask the user to select which stress components to plot
    print("Select which stress components to plot:")
    print("0: σ_x, 1: σ_y, 2: σ_z, 3: σ_xy, 4: σ_xz, 5: σ_yz")
    selected_indices = input("Enter the indices of the components to plot (e.g., 0, 1, 3): ").split(",")
    selected_indices = [int(i) for i in selected_indices]
    
    # Save directory for the plots
    save_directory = "./plots"
    os.makedirs(save_directory, exist_ok=True)  # Create the directory if it doesn't exist

    # Use get_next_save_path to get the next available file name
    base_filename = f"stress_{plane}_value_{value}"
    save_path =  "/workspaces/eia/data/output.png"
    
    # Plot the 2D results and save the plot
    plot_2d_results(x2_slice, y2_slice, stress_slice, selected_indices, plane, value, save_directory=save_directory, base_filename=base_filename)
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    main()
