import os
import numpy as np

STATES_DIR = "./final-data/states/"      # contains joint(6)+gripper(1)
EEF_DIR = "./final-data/eef-pos/"        # contains eef pose (6)

FINAL_STATES_DIR = "./final-data/final-states"
FINAL_ACTIONS_DIR = "./final-data/final-actions"

os.makedirs(FINAL_STATES_DIR, exist_ok=True)
os.makedirs(FINAL_ACTIONS_DIR, exist_ok=True)

def compute_deltas(arr):
    """Compute row-wise differences."""
    d = np.zeros_like(arr)
    d[1:] = arr[1:] - arr[:-1]
    return d

for fname in sorted(os.listdir(STATES_DIR)):
    # Skip if not a .txt OR if contains 'bkp'
    if (not fname.endswith(".txt")) or ("bkp" in fname.lower()):
        continue

    print("Processing:", fname)

    # ------------ Load data ------------
    states = np.loadtxt(os.path.join(STATES_DIR, fname))  # (N, 7)
    eef = np.loadtxt(os.path.join(EEF_DIR, fname))        # (N, 6)

    assert states.shape[0] == eef.shape[0], "Frame count mismatch!"

    N = states.shape[0]

    # ------------ Build final-states ------------
    # shape (N, 13)
    final_states = np.hstack([eef, states])  # [6 eef, 7 state]

    # ------------ Build final-actions ------------
    final_actions = compute_deltas(final_states)  # (N, 13)

    # ------------ Save ------------
    np.savetxt(os.path.join(FINAL_STATES_DIR, fname), final_states, fmt="%.6f")
    np.savetxt(os.path.join(FINAL_ACTIONS_DIR, fname), final_actions, fmt="%.6f")

print("\n✔ Done — final-states and final-actions created.")
