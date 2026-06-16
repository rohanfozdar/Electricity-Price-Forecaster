from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"

# ERCOT load zones: name -> (latitude, longitude)
ERCOT_LOAD_ZONES = {
    "Houston": (29.76, -95.37),
    "North (Dallas)": (32.78, -96.80),
    "South (San Antonio)": (29.42, -98.49),
    "West (Midland)": (31.99, -102.08),
}

# Date range for historical data
DATA_START = "2016-01-01"
DATA_END = "2024-12-31"

# Reddit pipeline was permanently dropped — no API key required
# (sentiment signal was too weak and Reddit rate-limits made it unreliable)

# Price spike threshold in $/MWh
SPIKE_THRESHOLD_MWH = 200
