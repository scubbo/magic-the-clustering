"""
Downloads the Scryfall oracle_cards bulk data JSON to data/cards.json.
Fetches the current download URL from the Scryfall bulk-data API first,
so the local copy is always the latest version.
"""
import sys
from pathlib import Path

BULK_DATA_API = "https://api.scryfall.com/bulk-data"
DATA_DIR = Path(__file__).parent.parent / "data"

_HEADERS = {"User-Agent": "magic-the-clustering/0.1 (contact: scubbojj@gmail.com)"}


def _get_oracle_cards_url() -> str:
    import requests
    resp = requests.get(BULK_DATA_API, headers=_HEADERS)
    resp.raise_for_status()
    for entry in resp.json()["data"]:
        if entry["type"] == "oracle_cards":
            return entry["download_uri"]
    raise RuntimeError("Could not find oracle_cards entry in Scryfall bulk-data API response")


def download(dest: Path = DATA_DIR / "cards.json") -> Path:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Fetching current oracle_cards download URL...", flush=True)
    url = _get_oracle_cards_url()
    print(f"Downloading from {url} ...", flush=True)
    with requests.get(url, headers=_HEADERS, stream=True) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as out:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                out.write(chunk)
    print(f"Saved to {dest}", flush=True)
    return dest


if __name__ == "__main__":
    download()
