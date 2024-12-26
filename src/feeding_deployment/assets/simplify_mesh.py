import pybullet as p
import pybullet_data
import os

def simplify_mesh_with_pybullet(input_mesh_path, output_mesh_path, vhacd_params=None):
    """
    Simplifies a mesh using VHACD through PyBullet.

    Args:
        input_mesh_path (str): Path to the input mesh file (e.g., .obj).
        output_mesh_path (str): Path to save the simplified mesh.
        vhacd_params (dict, optional): Parameters for VHACD. Default uses PyBullet defaults.
    """
    if vhacd_params is None:
        vhacd_params = {
            "resolution": 100000,       # Maximum number of voxels
            "depth": 10,               # Maximum recursion depth
            "concavity": 0.0025,       # Concavity tolerance (lower = more detailed)
            "planeDownsampling": 4,    # Level of downsampling
            "convexhullDownsampling": 4,  # Convex hull downsampling
            "alpha": 0.05,             # Balance weight for concavity vs. volume preservation
            "beta": 0.05,              # Balance weight for concavity vs. surface area
            "maxNumVerticesPerCH": 64, # Limit for vertices per convex hull
            "minVolumePerCH": 0.0001   # Minimum volume per convex hull
        }

    if not os.path.exists(input_mesh_path):
        raise FileNotFoundError(f"Input mesh file not found: {input_mesh_path}")

    # Load PyBullet and configure search path
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Perform VHACD decomposition
    success = p.vhacd(input_mesh_path, output_mesh_path, "vhacd_log.txt")

    if success:
        print(f"Simplified mesh saved to: {output_mesh_path}")
    else:
        print("Failed to simplify the mesh.")

    # Disconnect PyBullet
    p.disconnect()

# Example usage:
input_mesh = "table/table.obj"
output_mesh = "table/simplified_table.obj"
simplify_mesh_with_pybullet(input_mesh, output_mesh)
