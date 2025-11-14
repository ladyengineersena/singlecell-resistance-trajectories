"""Temporary script to copy files from workspace to physical filesystem"""
import os
import shutil
from pathlib import Path

# List of files to copy (relative to workspace root)
files_to_copy = [
    # Scripts
    'scripts/generate_synthetic_sc.py',
    'scripts/run_full_pipeline.py',
    # Source files - will be added manually
]

# For now, just create the script
print("File copy script created. Manual file addition needed.")

