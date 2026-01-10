import numpy as np
import os
import shutil

path = r"C:\Users\stic\Downloads\gdghackathon-main\gdghackathon-main\smplx\SMPLX_NEUTRAL.npz"
backup_path = path + ".bak"

if not os.path.exists(path):
    print(f"File not found: {path}")
    exit(1)

# Backup
if not os.path.exists(backup_path):
    print(f"Backing up to {backup_path}")
    shutil.copy2(path, backup_path)
else:
    print(f"Backup already exists at {backup_path}")

try:
    data = np.load(path)
    data_dict = dict(data)
    
    print("Existing keys:", list(data_dict.keys()))

    # Keys that seem to be required by shattered smplx installation
    # Based on traceback: AttributeError: 'Struct' object has no attribute 'hands_componentsl'
    # And potentially others
    
    updates = {}
    
    # Dummy shape for PCA components: (num_pca, num_hand_params)
    # Standard SMPL-X hand PCA is usually 6, 12, or 45 components. 
    # The flat hand pose dim is 15 joints * 3 = 45.
    # We'll provide enough zeros.
    
    if 'hands_componentsl' not in data_dict:
        print("Adding dummy hands_componentsl")
        updates['hands_componentsl'] = np.zeros((45, 45), dtype=np.float32)
        
    if 'hands_componentsr' not in data_dict:
        print("Adding dummy hands_componentsr")
        updates['hands_componentsr'] = np.zeros((45, 45), dtype=np.float32)

    # Some versions check for 'hands_mean'
    if 'hands_meanl' not in data_dict:
        print("Adding dummy hands_meanl")
        updates['hands_meanl'] = np.zeros((45,), dtype=np.float32)

    if 'hands_meanr' not in data_dict:
        print("Adding dummy hands_meanr")
        updates['hands_meanr'] = np.zeros((45,), dtype=np.float32)

    if 'lmk_faces_idx' not in data_dict:
        print("Adding dummy lmk_faces_idx")
        # Ensure it has at least one element to be safe, or empty. 
        # Standard might be (51,) or similar.
        updates['lmk_faces_idx'] = np.zeros((0,), dtype=np.int64)
        
    # Face components (often 100 or 300)
    if 'face_components' not in data_dict:
        print("Adding dummy face_components")
        updates['face_components'] = np.zeros((300, 300), dtype=np.float32) # Enough for default
        
    data_dict.update(updates)
    
    # Save back
    if updates:
        print("Saving repaired file...")
        np.savez(path, **data_dict)
        print("Done.")
    else:
        print("No updates needed.")

except Exception as e:
    print(f"Error repairing: {e}")
    import traceback
    traceback.print_exc()
