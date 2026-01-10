import numpy as np
import os

path = r"C:\Users\stic\Downloads\gdghackathon-main\gdghackathon-main\smplx\SMPLX_NEUTRAL.npz"

if not os.path.exists(path):
    print(f"File not found: {path}")
    exit(1)

try:
    data = np.load(path)
    print("Keys in SMPLX_NEUTRAL.npz:")
    for k in data.keys():
        print(f" - {k}")

    # Check specifically for hands components
    print("\nCheck for hands_components:")
    if 'hands_componentsl' in data:
        print("FOUND: hands_componentsl")
    else:
        print("MISSING: hands_componentsl")

    if 'hands_components' in data:
        print("FOUND: hands_components")
    
    if 'hands_components_l' in data:
        print("FOUND: hands_components_l")

except Exception as e:
    print(f"Error loading: {e}")
