import numpy as np
import matplotlib.pyplot as plt
import os

def plot_gripper_data(folder_path):
    # File paths
    agent_pos_path = os.path.join(folder_path, "agent_pos.txt")
    actions_path = os.path.join(folder_path, "actions.txt")

    if not os.path.exists(agent_pos_path) or not os.path.exists(actions_path):
        print(f"Error: Could not find files in {folder_path}")
        return

    # Load data
    agent_pos = np.loadtxt(agent_pos_path)
    actions = np.loadtxt(actions_path)

    # Extract gripper columns
    gripper_state = agent_pos[:, 12]
    gripper_action = actions[:, 12]

    # Define Phase Regions (Based on your simulation steps)
    # Move(50) + Down(50) = 100. Close(25). Lift(50) + MoveTray(150) = 325. Open(25).
    close_region = (100, 125)
    open_region = (325, len(gripper_state))

    # Create Plot (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Gripper Phase Analysis: {os.path.basename(folder_path)}", fontsize=16, fontweight='bold')

    # Function to add highlight regions to an axis
    def highlight_phases(ax):
        ax.axvspan(close_region[0], close_region[1], color='green', alpha=0.2, label='Closing Phase')
        ax.axvspan(open_region[0], open_region[1], color='red', alpha=0.2, label='Opening Phase')

    # --- Plot 1: Gripper State ---
    ax1.plot(gripper_state, color='blue', linewidth=2.5, label='Normalized State (0-1)')
    highlight_phases(ax1)
    ax1.set_title("Robot State: Gripper Joint Angle", fontsize=14)
    ax1.set_ylabel("Normalized Position")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')

    # --- Plot 2: Gripper Action ---
    ax2.step(range(len(gripper_action)), gripper_action, where='post', color='darkorange', linewidth=2.5, label='Discrete Command')
    highlight_phases(ax2)
    ax2.set_title("Robot Action: Gripper Command Mapping", fontsize=14)
    ax2.set_xlabel("Frame Index", fontsize=12)
    ax2.set_ylabel("Action Value (-1, 0, 1)")
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_yticks([-1, 0, 1])
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # Update this path to your specific dataset folder
    iteration_folder = "./dataset/iter_0000" 
    plot_gripper_data(iteration_folder)