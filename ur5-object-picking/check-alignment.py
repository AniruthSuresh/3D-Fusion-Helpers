import os
import numpy as np
import cv2
import open3d as o3d

def load_point_clouds(iter_folder, frame_idx, version):
    """Load point clouds for a specific version (v1, v2, or v3)"""
    
    # Load third-person
    tp_pcd_path = os.path.join(iter_folder, "third_person", f"pcd_{version}", 
                                f"tp_pcd_{version}_{frame_idx:04d}.npy")
    tp_pcd = np.load(tp_pcd_path)
    
    tp_rgb_path = os.path.join(iter_folder, "third_person", "rgb", f"tp_rgb_{frame_idx:04d}.png")
    tp_rgb = cv2.imread(tp_rgb_path)
    tp_rgb = cv2.cvtColor(tp_rgb, cv2.COLOR_BGR2RGB)
    tp_colors = tp_rgb.reshape(-1, 3) / 255.0
    
    # Load wrist
    wr_pcd_path = os.path.join(iter_folder, "wrist", f"pcd_{version}", 
                                f"wr_pcd_{version}_{frame_idx:04d}.npy")
    wr_pcd = np.load(wr_pcd_path)
    
    wr_rgb_path = os.path.join(iter_folder, "wrist", "rgb", f"wr_rgb_{frame_idx:04d}.png")
    wr_rgb = cv2.imread(wr_rgb_path)
    wr_rgb = cv2.cvtColor(wr_rgb, cv2.COLOR_BGR2RGB)
    wr_colors = wr_rgb.reshape(-1, 3) / 255.0
    
    return tp_pcd, tp_colors, wr_pcd, wr_colors

def filter_point_cloud(points, colors, z_min=-0.5, z_max=2.5):
    """Filter points by z-range"""
    valid_mask = (points[:, 2] > z_min) & (points[:, 2] < z_max)
    return points[valid_mask], colors[valid_mask]

def analyze_point_cloud(name, points):
    """Print statistics about a point cloud"""
    print(f"\n{name}:")
    print(f"  Shape: {points.shape}")
    print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    print(f"  Centroid: [{points[:, 0].mean():.3f}, {points[:, 1].mean():.3f}, {points[:, 2].mean():.3f}]")
    
    # Sample some points
    print(f"  Sample points:")
    for i in range(min(3, len(points))):
        print(f"    [{points[i, 0]:.3f}, {points[i, 1]:.3f}, {points[i, 2]:.3f}]")

def create_colored_point_cloud(points, colors, uniform_color=None):
    """Create Open3D point cloud with colors"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if uniform_color is not None:
        pcd.paint_uniform_color(uniform_color)
    else:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def visualize_alignment(tp_pcd, wr_pcd, version_name):
    """Visualize alignment with color coding"""
    
    # Color third-person blue, wrist red
    tp_viz = o3d.geometry.PointCloud(tp_pcd)
    tp_viz.paint_uniform_color([0, 0, 1])  # Blue
    
    wr_viz = o3d.geometry.PointCloud(wr_pcd)
    wr_viz.paint_uniform_color([1, 0, 0])  # Red
    
    combined = tp_viz + wr_viz
    
    # Add coordinate frame at origin
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )
    
    print(f"\n{'='*60}")
    print(f"Visualizing {version_name}")
    print(f"{'='*60}")
    print("Color Legend:")
    print("  BLUE = Third-person camera")
    print("  RED = Wrist camera")
    print("  PURPLE/MAGENTA = Good overlap (aligned)")
    print("\nLook for:")
    print("  ✓ Purple/magenta colors where cameras overlap")
    print("  ✓ Table/plane at z ≈ 0.6")
    print("  ✓ Robot gripper visible in both views")
    print("  ✗ Completely separate blue/red clouds = NOT aligned")
    
    o3d.visualization.draw_geometries(
        [combined, coordinate_frame],
        window_name=f"Alignment Test - {version_name}",
        width=1280,
        height=720
    )

def compute_alignment_score(tp_pcd, wr_pcd):
    """
    Compute a simple alignment score based on centroid distance
    Lower is better (should be close if aligned)
    """
    tp_centroid = tp_pcd.mean(axis=0)
    wr_centroid = wr_pcd.mean(axis=0)
    
    distance = np.linalg.norm(tp_centroid - wr_centroid)
    
    return distance

def test_version(iter_folder, frame_idx, version):
    """Test a specific version"""
    print(f"\n{'='*60}")
    print(f"TESTING VERSION: {version.upper()}")
    print(f"{'='*60}")
    
    # Load data
    tp_pcd, tp_colors, wr_pcd, wr_colors = load_point_clouds(iter_folder, frame_idx, version)
    
    # Filter
    tp_pcd_filt, tp_colors_filt = filter_point_cloud(tp_pcd, tp_colors)
    wr_pcd_filt, wr_colors_filt = filter_point_cloud(wr_pcd, wr_colors)
    
    # Analyze
    analyze_point_cloud(f"Third-person ({version})", tp_pcd_filt)
    analyze_point_cloud(f"Wrist ({version})", wr_pcd_filt)
    
    # Compute alignment score
    alignment_score = compute_alignment_score(tp_pcd_filt, wr_pcd_filt)
    print(f"\nAlignment Score (centroid distance): {alignment_score:.3f}")
    print(f"  (Lower is better, should be < 0.5 if well aligned)")
    
    # Create Open3D point clouds
    tp_o3d = create_colored_point_cloud(tp_pcd_filt, tp_colors_filt)
    wr_o3d = create_colored_point_cloud(wr_pcd_filt, wr_colors_filt)
    
    # Visualize
    visualize_alignment(tp_o3d, wr_o3d, version)
    
    return alignment_score

def compare_all_versions(iter_folder, frame_idx=0):
    """Test all three versions and compare"""
    
    print(f"\n{'#'*60}")
    print(f"POINT CLOUD ALIGNMENT TEST")
    print(f"Testing iteration: {iter_folder}")
    print(f"Frame index: {frame_idx}")
    print(f"{'#'*60}")
    
    scores = {}
    
    for version in ['v1', 'v2', 'v3']:
        try:
            score = test_version(iter_folder, frame_idx, version)
            scores[version] = score
        except Exception as e:
            print(f"\nERROR testing {version}: {e}")
            scores[version] = float('inf')
    
    # Summary
    print(f"\n{'='*60}")
    print("ALIGNMENT SUMMARY")
    print(f"{'='*60}")
    
    for version in ['v1', 'v2', 'v3']:
        score = scores[version]
        status = "✓ GOOD" if score < 0.5 else "✗ BAD"
        print(f"{version.upper()}: {score:.3f} {status}")
    
    best_version = min(scores, key=scores.get)
    print(f"\n🏆 BEST VERSION: {best_version.upper()} (score: {scores[best_version]:.3f})")
    print(f"\nRECOMMENDATION:")
    print(f"Use depth_to_point_cloud_{best_version} in your simulation code")
    
    return best_version, scores

def visualize_combined_with_original_colors(iter_folder, frame_idx, version):
    """Visualize combined point cloud with original RGB colors"""
    
    print(f"\nVisualizing {version} with original colors...")
    
    tp_pcd, tp_colors, wr_pcd, wr_colors = load_point_clouds(iter_folder, frame_idx, version)
    
    tp_pcd_filt, tp_colors_filt = filter_point_cloud(tp_pcd, tp_colors)
    wr_pcd_filt, wr_colors_filt = filter_point_cloud(wr_pcd, wr_colors)
    
    # Create point clouds with original colors
    tp_o3d = create_colored_point_cloud(tp_pcd_filt, tp_colors_filt)
    wr_o3d = create_colored_point_cloud(wr_pcd_filt, wr_colors_filt)
    
    combined = tp_o3d + wr_o3d
    combined = combined.voxel_down_sample(voxel_size=0.003)
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.3, origin=[0, 0, 0]
    )
    
    print("Showing combined point cloud with original RGB colors")
    o3d.visualization.draw_geometries(
        [combined, coordinate_frame],
        window_name=f"Combined Point Cloud - {version} (Original Colors)",
        width=1280,
        height=720
    )

def main():
    # Configuration
    iter_folder = "dataset/iter_0000"
    frame_idx = 0
    
    # Test all versions
    best_version, scores = compare_all_versions(iter_folder, frame_idx)
    
    # Show best version with original colors
    response = input(f"\nShow {best_version} with original RGB colors? (y/n): ")
    if response.lower() == 'y':
        visualize_combined_with_original_colors(iter_folder, frame_idx, best_version)
    
    # Option to test specific frame
    while True:
        response = input("\nTest another frame? Enter frame number or 'q' to quit: ")
        if response.lower() == 'q':
            break
        try:
            frame_num = int(response)
            test_version(iter_folder, frame_num, best_version)
        except:
            print("Invalid input")

if __name__ == "__main__":
    main()