import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

def extract_3d_slice(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a 3D field for visualization.

    :param x1: Meshgrid of x1 coordinates.
    :param x2: Meshgrid of x2 coordinates.
    :param x3: Meshgrid of x3 coordinates.
    :param field: 3D array of the field (stress, strain, etc.).
    :return: 3D arrays of x, y, z, and the field values.
    """
    return x1, x2, x3, field

def plot_3d_results(x, y, z, stress_component, component_index, save_path=None):
    """
    Plot the selected stress component in a 3D figure.

    :param x: 3D array of x coordinates.
    :param y: 3D array of y coordinates.
    :param z: 3D array of z coordinates.
    :param stress_component: 3D array of the stress component values to plot.
    :param component_index: Index specifying which stress component to plot (0-5).
    :param save_path: If provided, the plot will be saved to this path.
    """
    component_titles = ['Stress Component σ_x', 'Stress Component σ_y', 'Stress Component σ_z',
                        'Shear Stress Component σ_xy', 'Shear Stress Component σ_xz', 'Shear Stress Component σ_yz']

    selected_title = component_titles[component_index]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    c = ax.scatter(x, y, z, c=stress_component.flatten(), cmap='viridis')
    fig.colorbar(c, ax=ax)
    ax.set_title(selected_title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)  # Save the plot to the provided path
    else:
        plt.show()  # Display the plot if no save path is provided

def get_next_save_path(directory, base_filename):
    """
    Generate the next available save path in the directory for the given base filename.

    :param directory: Directory to save files in.
    :param base_filename: Base filename to use for numbering.
    :return: Full path for the next available file.
    """
    i = 0
    while True:
        save_path = os.path.join(directory, f"{base_filename}_{i}.png")
        if not os.path.exists(save_path):
            return save_path
        i += 1

# Example usage
if __name__ == "__main__":
    # Example data for demonstration purposes
    x = np.linspace(0, 10, 30)
    x1, x2, x3 = np.meshgrid(x, x, x, indexing='ij')
    field = np.random.rand(30, 30, 30, 6)  # Random field with 6 components (σ_x, σ_y, σ_z, σ_xy, σ_xz, σ_yz)

    # Extracting a 3D slice for a single component
    component_index = 0  # Index for σ_x
    x, y, z, field_3d = extract_3d_slice(x1, x2, x3, field[:, :, :, component_index])

    # Setting up directories and saving the plot
    directory = "data"
    base_filename = "output_3d"
    os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

    # Get the next available save path and plot
    save_path = get_next_save_path(directory, base_filename)
    plot_3d_results(x, y, z, field_3d, component_index, save_path=save_path)
