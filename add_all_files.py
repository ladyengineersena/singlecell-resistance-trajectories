"""Script to add all remaining files from workspace to physical filesystem"""
import os
import json
from pathlib import Path

# Base directory
base_dir = Path(__file__).parent

# Files to create (content will be read from workspace)
files_to_create = {
    # Source files
    'src/preprocess/qc.py': 'already_written',
    'src/preprocess/batch_correction.py': 'already_written',
    'src/trajectory/scvelo_wrapper.py': 'need_to_write',
    'src/features/feature_builder.py': 'need_to_write',
    'src/models/xgb_model.py': 'need_to_write',
    'src/models/mil_model.py': 'need_to_write',
    'src/models/gnns/gnn_model.py': 'need_to_write',
    'src/evaluate/metrics.py': 'need_to_write',
    'src/utils.py': 'already_written',
    
    # __init__ files
    'src/__init__.py': 'need_to_write',
    'src/preprocess/__init__.py': 'need_to_write',
    'src/trajectory/__init__.py': 'need_to_write',
    'src/features/__init__.py': 'need_to_write',
    'src/models/__init__.py': 'need_to_write',
    'src/models/gnns/__init__.py': 'need_to_write',
    'src/evaluate/__init__.py': 'need_to_write',
    
    # Data README
    'data/README_DATA.md': 'need_to_write',
}

print("File creation script ready. Files will be written individually.")

