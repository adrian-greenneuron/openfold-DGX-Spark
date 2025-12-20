#!/usr/bin/env python3
"""
Patch DeepSpeed for Blackwell GPU (sm_120/sm_121) compatibility.

This script patches DeepSpeed's builder.py to:
1. Clean up architecture strings (remove '+PTX', '.')
2. Map sm_121 to sm_120 (Blackwell variants)

Works with DeepSpeed 0.15.x and Python 3.10+
"""

import os
import sys
import glob


def find_deepspeed_builder():
    """Find DeepSpeed builder.py dynamically across Python versions."""
    # Common paths to search
    search_patterns = [
        "/usr/local/lib/python*/dist-packages/deepspeed/ops/op_builder/builder.py",
        "/usr/local/lib/python*/site-packages/deepspeed/ops/op_builder/builder.py",
        "/opt/conda/lib/python*/site-packages/deepspeed/ops/op_builder/builder.py",
    ]
    
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    
    # Fallback: try to find via deepspeed import
    try:
        import deepspeed
        ds_path = os.path.dirname(deepspeed.__file__)
        builder_path = os.path.join(ds_path, "ops", "op_builder", "builder.py")
        if os.path.exists(builder_path):
            return builder_path
    except ImportError:
        pass
    
    return None


def patch_builder(target_path):
    """Apply the Blackwell compatibility patch to builder.py."""
    print(f"Patching {target_path}...")
    
    with open(target_path, 'r') as f:
        content = f.read()
    
    # The code we want to replace
    old_code = 'num = cc[0] + cc[2]'
    
    # New code that handles Blackwell architecture
    # 1. Clean up the string (remove +PTX, dots)
    # 2. Map 121 -> 120 (sm_121 is Blackwell variant, use sm_120)
    patch_code = '''num = cc.replace("+PTX","").replace(".","")
            if num == "121": num="120"'''
    
    if old_code not in content:
        print(f"WARNING: Expected code pattern not found in builder.py")
        print(f"Looking for: '{old_code}'")
        print(f"This may indicate a different DeepSpeed version.")
        print(f"File location: {target_path}")
        return False
    
    new_content = content.replace(old_code, patch_code)
    
    with open(target_path, 'w') as f:
        f.write(new_content)
    
    print("SUCCESS: Patched DeepSpeed builder.py for Blackwell (sm_120/sm_121) compatibility")
    return True


def main():
    target = find_deepspeed_builder()
    
    if not target:
        print("ERROR: Could not find DeepSpeed builder.py")
        print("Searched common Python paths. Is DeepSpeed installed?")
        sys.exit(1)
    
    if not os.path.exists(target):
        print(f"ERROR: {target} not found!")
        sys.exit(1)
    
    success = patch_builder(target)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
