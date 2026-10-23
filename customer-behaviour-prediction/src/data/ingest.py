from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests
import yaml

################################################################################################
# -------------------------------- downloading dataset  ----------------------------------------
################################################################################################
def get_project_root() -> Path:
    """Return the root directory of the project."""
    return Path(__file__).resolve().parents[2]

def load_config() -> dict:
    """Load project configuration from config.yaml."""
    project_root = get_project_root()
    config_path = project_root / "config" / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

def download_online_retail_ii() -> Path:
    """
    Download Online Retail II from the UCI Machine Learning Repository
    and extract the Excel file into data/raw/.
    """
    config = load_config()

    project_root = get_project_root()

    raw_dir = project_root / config["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    filename = config["data"]["online_retail_ii"]["filename"]
    download_url = config["data"]["online_retail_ii"]["download_url"]

    output_file = raw_dir / filename

    # Don't download again if the file already exists
    if output_file.exists():
        print(f"Dataset already exists: {output_file}")
        return output_file

    zip_file = raw_dir / "online_retail_ii.zip"

    print("Downloading Online Retail II from UCI...")

    response = requests.get(download_url, stream=True, timeout=120)
    response.raise_for_status()

    with open(zip_file, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)

    print("Download completed.")
    print("Extracting dataset...")

    with ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    print(f"Dataset available at: {output_file}")

    return output_file


def load_online_retail_ii() -> pd.DataFrame:
    """
    Download if necessary and load Online Retail II
    into a Pandas DataFrame.
    """
    dataset_path = download_online_retail_ii()

    df_2009_2010 = pd.read_excel(dataset_path, sheet_name="Year 2009-2010")
    df_2010_2011 = pd.read_excel(dataset_path, sheet_name="Year 2010-2011")

    df = pd.concat([df_2009_2010, df_2010_2011], ignore_index=True)

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.sort_values("InvoiceDate").reset_index(drop=True)

    return df


def save_dataframe(
    df: pd.DataFrame,
    directory: str,
    filename: str,
) -> Path:
    """
    Save a dataframe to a project-relative directory.
    """

    project_root = get_project_root()

    output_directory = project_root / directory
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = output_directory / filename

    if output_path.suffix.lower() != ".csv":
        raise ValueError("Only .csv files are supported.")

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"Dataframe saved successfully: {output_path}")

    return output_path


def load_dataframe(
    directory: str,
    filename: str,
) -> pd.DataFrame:
    """
    Load a CSV dataframe from a project-relative directory.

    Parameters
    ----------
    directory : str
        Project-relative directory containing the dataset.
    filename : str
        Name of the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """

    project_root = get_project_root()

    input_path = project_root / directory / filename

    if not input_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {input_path}"
        )

    if input_path.suffix.lower() != ".csv":
        raise ValueError(
            "Only .csv files are supported."
        )

    df = pd.read_csv(input_path)

    print(f"Dataframe loaded successfully: {input_path}")
    print(f"Shape: {df.shape}")

    return df
