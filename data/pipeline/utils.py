import os
import pandas as pd
import json

def save_data_to_csv(data_df: pd.DataFrame, filename: str, output_dir: str = "data/output"):
    """
    Save dataframe to CSV file
    
    Args:
        data_df (pd.DataFrame): The dataframe to save
        filename (str): The name of the file to save
        output_dir (str): The directory to save the file to
        
    Returns:
        str: The path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    data_df.to_csv(output_path, index=False)
    print(f"Saved {len(data_df)} records to {output_path}")
    return output_path

def save_data_to_json(data_df: pd.DataFrame, filename: str, output_dir: str = "data/output"):
    """
    Save dataframe to JSON file
    
    Args:
        data_df (pd.DataFrame): The dataframe to save
        filename (str): The name of the file to save
        output_dir (str): The directory to save the file to
        
    Returns:
        str: The path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    # Convert DataFrame to list of dictionaries
    records = data_df.to_dict(orient="records")
    
    with open(output_path, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"Saved {len(records)} records to {output_path}")
    return output_path 