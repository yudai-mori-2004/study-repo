#!/usr/bin/env python3
"""
HDF5 file loading utility
"""

import os
import h5py
import numpy as np
import csv
from typing import Union, Optional, Dict, List


def load_h5_data(file_path: str, dataset_name: Optional[str] = None) -> np.ndarray:
    """
    Load data from HDF5 file and return as numpy array
    
    Args:
        file_path (str): Path to HDF5 file
        dataset_name (str, optional): Dataset name. If not specified, use first dataset
        
    Returns:
        np.ndarray: n-dimensional numpy array
        
    Raises:
        FileNotFoundError: If file does not exist
        KeyError: If specified dataset name does not exist
        ValueError: If file is invalid
    """
    
    # File existence check
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")
    
    if not file_path.endswith('.h5'):
        raise ValueError(f"Not an HDF5 file: {file_path}")
    
    try:
        with h5py.File(file_path, "r") as f:
            # Use first dataset if dataset_name is not specified
            if dataset_name is None:
                if len(f.keys()) == 0:
                    raise ValueError(f"No datasets found in HDF5 file: {file_path}")
                dataset_name = list(f.keys())[0]
            
            # Dataset existence check
            if dataset_name not in f:
                available_datasets = list(f.keys())
                raise KeyError(f"Dataset '{dataset_name}' not found. Available datasets: {available_datasets}")
            
            # Load data
            dataset = f[dataset_name]
            data = dataset[()]  # Load all data at once
            
            # Return as numpy array
            return np.array(data)
            
    except Exception as e:
        raise ValueError(f"Failed to load HDF5 file: {file_path}, Error: {str(e)}")


def inspect_h5_file(file_path: str) -> dict:
    """
    Get structure and basic information of HDF5 file
    
    Args:
        file_path (str): Path to HDF5 file
        
    Returns:
        dict: Dictionary containing file information
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")
    
    info = {
        'file_path': file_path,
        'datasets': {}
    }
    
    try:
        with h5py.File(file_path, "r") as f:
            for dataset_name, dataset in f.items():
                data = dataset[()]
                info['datasets'][dataset_name] = {
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'size': data.size,
                    'ndim': data.ndim,
                    'attributes': dict(dataset.attrs.items()) if hasattr(dataset, 'attrs') else {}
                }
                
    except Exception as e:
        raise ValueError(f"Failed to inspect HDF5 file: {file_path}, Error: {str(e)}")
    
    return info


def load_multiple_h5_files(file_paths: list, dataset_name: Optional[str] = None) -> list:
    """
    Load data from multiple HDF5 files
    
    Args:
        file_paths (list): List of HDF5 file paths
        dataset_name (str, optional): Dataset name
        
    Returns:
        list: List of numpy arrays
    """
    
    results = []
    for file_path in file_paths:
        try:
            data = load_h5_data(file_path, dataset_name)
            results.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {str(e)}")
            results.append(None)
    
    return results


def load_csv_data(file_path: str, encoding: str = 'utf-8') -> List[Dict[str, str]]:
    """
    Load CSV file and return as list of dictionaries
    
    Args:
        file_path (str): Path to CSV file
        encoding (str): File encoding (default: utf-8)
        
    Returns:
        List[Dict[str, str]]: List of dictionaries where keys are column headers
        
    Raises:
        FileNotFoundError: If file does not exist
        ValueError: If file is invalid or encoding error occurs
    """
    
    # File existence check
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    if not file_path.endswith('.csv'):
        raise ValueError(f"Not a CSV file: {file_path}")
    
    try:
        data = []
        with open(file_path, 'r', encoding=encoding, newline='') as csvfile:
            # Use DictReader to automatically handle headers
            reader = csv.DictReader(csvfile)
            
            # Check if file has headers
            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no headers: {file_path}")
            
            # Load all rows
            for row in reader:
                data.append(dict(row))
                
        return data
        
    except UnicodeDecodeError as e:
        raise ValueError(f"Encoding error in CSV file: {file_path}. Try different encoding. Error: {str(e)}")
    except Exception as e:
        raise ValueError(f"Failed to load CSV file: {file_path}, Error: {str(e)}")


def inspect_csv_file(file_path: str, encoding: str = 'utf-8') -> Dict:
    """
    Get structure and basic information of CSV file
    
    Args:
        file_path (str): Path to CSV file
        encoding (str): File encoding (default: utf-8)
        
    Returns:
        Dict: Dictionary containing file information
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    info = {
        'file_path': file_path,
        'encoding': encoding,
        'headers': [],
        'num_rows': 0,
        'num_columns': 0
    }
    
    try:
        with open(file_path, 'r', encoding=encoding, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Get headers
            info['headers'] = list(reader.fieldnames) if reader.fieldnames else []
            info['num_columns'] = len(info['headers'])
            
            # Count rows
            for row in reader:
                info['num_rows'] += 1
                
    except Exception as e:
        raise ValueError(f"Failed to inspect CSV file: {file_path}, Error: {str(e)}")
    
    return info


if __name__ == "__main__":
    # Test sample code
    sample_dir = "../sample"
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.h5')]
        
        for sample_file in sample_files[:2]:  # Test first 2 files
            file_path = os.path.join(sample_dir, sample_file)
            print(f"\n=== Test: {sample_file} ===")
            
            try:
                # Get file information
                info = inspect_h5_file(file_path)
                print(f"Number of datasets: {len(info['datasets'])}")
                
                for ds_name, ds_info in info['datasets'].items():
                    print(f"- {ds_name}: shape={ds_info['shape']}, dtype={ds_info['dtype']}")
                
                # Load data
                data = load_h5_data(file_path)
                print(f"Load successful: shape={data.shape}, dtype={data.dtype}")
                
            except Exception as e:
                print(f"Error: {str(e)}")