"""Download and load the IBM Telco Customer Churn sample dataset."""
from pathlib import Path
from urllib.request import urlopen
import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"


def download_data(force: bool = False) -> Path:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        print(f"Dataset already exists: {DATA_PATH}")
        return DATA_PATH
    print("Downloading IBM Telco Customer Churn dataset...")
    with urlopen(DATA_URL, timeout=30) as response:
        content = response.read()
    if len(content) < 100_000:
        raise RuntimeError("Downloaded file is unexpectedly small; check the source URL.")
    DATA_PATH.write_bytes(content)
    print(f"Saved {len(content):,} bytes to {DATA_PATH}")
    return DATA_PATH


def load_data() -> pd.DataFrame:
    path = download_data()
    df = pd.read_csv(path)
    if "Churn" not in df.columns:
        raise ValueError("Dataset does not contain the expected 'Churn' target column.")
    return df


if __name__ == "__main__":
    data = load_data()
    print(f"Shape: {data.shape}")
    print(data.head())
    print("\nTarget distribution:")
    print(data["Churn"].value_counts(dropna=False))
