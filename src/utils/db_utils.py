"""
Database utilities for RACECAR proxy analysis.

This module provides simple functions to discover and load .db3 files.
"""

from pathlib import Path


def find_all_db3_files(root_path=None, exclude_patterns=None):
    """
    Recursively find all .db3 database files in the workspace.
    
    Args:
        root_path (Path or str): Root directory to search. If None, searches from the 
                                  racecar-energy-proxy/data folder.
        exclude_patterns (list): Folder/file patterns to exclude (e.g., ['S1', 'S2'])
    
    Returns:
        list: Sorted list of Path objects pointing to all .db3 files found.
    
    Examples:
        >>> db_files = find_all_db3_files()  # Searches from racecar-energy-proxy/data
        >>> db_files = find_all_db3_files(root_path="/path/to/workspace")
        >>> db_files = find_all_db3_files(exclude_patterns=['S1', 'test'])
    """
    if root_path is None:
        # Get the absolute path of this file, then navigate to data folder
        current_file = Path(__file__).resolve()
        # Path: .../racecar-energy-proxy/src/utils/db_utils.py
        root_path = current_file.parent.parent.parent / "data"
    else:
        root_path = Path(root_path).resolve()
    
    if exclude_patterns is None:
        exclude_patterns = []
    
    # Verify the path exists
    if not root_path.exists():
        print(f"Warning: Search path does not exist: {root_path}")
        return []
    
    # Find all .db3 files recursively
    all_db3_files = list(root_path.glob("**/*.db3"))
    
    # Filter out excluded patterns
    if exclude_patterns:
        filtered_files = []
        for f in all_db3_files:
            path_str = str(f)
            if not any(pattern in path_str for pattern in exclude_patterns):
                filtered_files.append(f)
        all_db3_files = filtered_files
    
    return sorted(all_db3_files)
