import os
import pandas as pd
import json

from data.settings import OUTPUT_PATH

def _save_json(data: dict | list, output_path: str, filename: str) -> str:
    """
    Helper function to save data to JSON file
    
    Args:
        data: The data to save (dict or list)
        output_path: The directory to save the file to
        filename: The name of the file to save
        
    Returns:
        str: The path to the saved file
    """
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, filename)
    
    try:
        with open(full_path, 'w') as f:
            json.dump(data, f, indent=2)
    except IOError as e:
        print(f"Error saving file {full_path}: {e}")
        raise
        
    print(f"Saved {len(data)} records to {full_path}")
    return full_path

def save_data_to_json(data_df: pd.DataFrame, output_path: str = OUTPUT_PATH, filename: str = "personas.json") -> str:
    """
    Save dataframe to JSON file
    
    Args:
        data_df: The dataframe to save
        output_path: The directory to save the file to
        filename: The name of the file to save
        
    Returns:
        str: The path to the saved file
    """
    records = data_df.to_dict(orient="records")
    return _save_json(records, output_path, filename)

def save_dict_to_json(filename: str, data_dict: dict, output_path: str = OUTPUT_PATH) -> str:
    """
    Save dictionary to JSON file
    
    Args:
        filename: The name of the file to save
        data_dict: The dictionary to save
        output_path: The directory to save the file to
        
    Returns:
        str: The path to the saved file
    """
    return _save_json(data_dict, output_path, filename)
