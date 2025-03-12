# data_utils.py
import pandas as pd
import glob

def load_data_from_directory(directory, pattern="*_fusions.tsv"):
    """
    Load and concatenate files matching a pattern in a directory.
    
    Parameters:
        directory (str): The root directory where the files are located.
        pattern (str): The glob pattern to match file names (default: "*_fusions.tsv").
        
    Returns:
        pandas.DataFrame: A concatenated DataFrame of all matched files.
    """
    # Recursively find all files matching the pattern in the directory
    file_paths = glob.glob(f"{directory}/**/{pattern}", recursive=True)
    # Read each file as a DataFrame and collect them into a list
    dfs = [pd.read_csv(fp, sep='\t') for fp in file_paths]
    # Concatenate all DataFrames and return the result
    return pd.concat(dfs, ignore_index=True)

def preprocess_dataframe(df):
    """
    Perform common preprocessing tasks on a DataFrame.
    
    This function is intended to serve as a starting point for your
    custom preprocessing. It currently removes duplicate rows.
    
    Parameters:
        df (pandas.DataFrame): The DataFrame to preprocess.
        
    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
    """
    # Drop duplicate rows to avoid redundant data
    df = df.drop_duplicates()
    # Add any additional preprocessing steps here (e.g., renaming columns,
    # handling missing values, filtering rows, etc.)
    return df
