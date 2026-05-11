"""
One-time data download script.

Run this script once to download all market and macro data from
Yahoo Finance and FRED, saving the results to CSV files.
The main pipeline (main.py) reads from these CSV files so that
results are fully reproducible without an internet connection.

Usage
-----
    python download_data.py
"""

import sys
import os

# Make sure 'code/' is on the path when running from the project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import DataLoader
from src.config import MARKET_DATA_CSV, FRED_DATA_CSV, DATA_DIR


def main():
    print("=" * 60)
    print("  EC581 — Data Download")
    print("=" * 60)
    print(f"  Saving to: {DATA_DIR}")
    print()

    loader = DataLoader()
    loader.download_and_save()

    print()
    print("  Done! Files created:")
    print(f"    {MARKET_DATA_CSV}")
    print(f"    {FRED_DATA_CSV}")
    print()
    print("  You can now run: python main.py")


if __name__ == "__main__":
    main()
