"""
Downloads the Scryfall oracle_cards bulk data JSON to data/cards.json.
Fetches the current download URL from the Scryfall bulk-data API first,
so the local copy is always the latest version.
"""
import json
import sys
import urllib.request
from pathlib import Path

BULK_DATA_API = "https://api.scryfall.com/bulk-data"
DATA_DIR = Path(__file__).parent.parent / "data"


def _get_oracle_cards_url() -> str:
    with urllib.request.urlopen(BULK_DATA_API) as resp:
        payload = json.load(resp)
    for entry in payload["data"]:
        if entry["type"] == "oracle_cards":
            return entry["download_uri"]
    raise RuntimeError("Could not find oracle_cards entry in Scryfall bulk-data API response")


def download(dest: Path = DATA_DIR / "cards.json") -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Fetching current oracle_cards download URL...", flush=True)
    url = _get_oracle_cards_url()
    print(f"Downloading from {url} ...", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}", flush=True)
    return dest


if __name__ == "__main__":
    download()
