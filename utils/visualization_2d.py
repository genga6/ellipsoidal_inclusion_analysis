import matplotlib.pyplot as plt
import numpy as np
import os

def extract_2d_slice(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, field: np.ndarray, plane: str, value: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract a 2D slice from a 3D field along a specified plane and value.

    :param x1: Meshgrid of x1 coordinates.
    :param x2: Meshgrid of x2 coordinates.
    :param x3: Meshgrid of x3 coordinates.
    :param field: 4D array of the field (e.g., stress, strain, etc.).
    :param plane: Plane to extract ('x1', 'x2', or 'x3').
    :param value: Value of the plane coordinate.
    :return: 2D arrays of x, y, and the field values.
    """
    if plane == "x3":
        idx = np.argmin(np.abs(x3[0, 0, :] - value))  # Find the closest index to the value along x3
        x, y = x1[:, :, idx], x2[:, :, idx]
        field_2d = field[:, :, idx, 0]  # Extract the first stress component (e.g., σxx)
    elif plane == "x2":
        idx = np.argmin(np.abs(x2[0, :, 0] - value))  # Find the closest index to the value along x2
        x, y = x1[:, idx, :], x3[:, idx, :]
        field_2d = field[:, idx, :, 0]  # Extract the first stress component (e.g., σxx)
    elif plane == "x1":
        idx = np.argmin(np.abs(x1[:, 0, 0] - value))  # Find the closest index to the value along x1
        x, y = x2[:, 0, :], x3[:, 0, :]
        field_2d = field[idx, :, :, 0]  # Extract the first stress component (e.g., σxx)
    
    return x, y, field_2d




def plot_2d_results(x: np.ndarray, y: np.ndarray, field_2d: np.ndarray, selected_indices: list, plane: str, value: float, save_directory: str, base_filename: str):
    """
    Plot 2D contour of selected stress or strain components.

    :param x: 2D array of x coordinates.
    :param y: 2D array of y coordinates.
    :param field_2d: 2D array of the field (stress, strain, etc.) to be plotted.
    :param selected_indices: List of indices to select specific stress/strain components.
    :param plane: Plane ('x1', 'x2', 'x3') used for slicing.
    :param value: Value of the plane coordinate.
    :param save_directory: Directory to save the plot.
    :param base_filename: Base filename to be used for generating the save path.
    """
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot contourf for the selected stress/strain components
    contour = ax.contourf(x, y, field_2d, cmap='viridis', levels=50)
    
    # Add a colorbar
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Field Value')

    # Set labels and title
    ax.set_xlabel(f'{plane} coordinate')
    ax.set_ylabel(f'{plane} coordinate')
    ax.set_title(f'{plane} plane at {value} - Selected Components')

    # Generate the next available save path
    save_path = get_next_save_path(save_directory, base_filename)
    
    # Save the plot
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved at: {save_path}")

    # Show the plot (optional)
    plt.show()



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