import shutil
import urllib.request
import zipfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
DATA_URL = "https://huggingface.co/datasets/chitradrishti/fer2013/resolve/main/fer2013.csv.zip"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = DATA_DIR / "fer2013.csv"
    if csv_path.exists():
        print(f"Dataset already exists: {csv_path}")
        return

    archive_path = DATA_DIR / "fer2013.csv.zip"
    if not archive_path.exists() or not zipfile.is_zipfile(archive_path):
        print("Downloading FER-2013...")
        partial_path = DATA_DIR / "fer2013.csv.zip.part"
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "FER-Lab/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            with partial_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        partial_path.replace(archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if Path(name).name == "fer2013.csv"]
        if len(members) != 1:
            raise ValueError("Expected exactly one fer2013.csv in the archive.")
        partial_csv = DATA_DIR / "fer2013.csv.part"
        with archive.open(members[0]) as source, partial_csv.open("wb") as output:
            shutil.copyfileobj(source, output)
        partial_csv.replace(csv_path)

    print(f"Dataset saved to: {csv_path}")


if __name__ == "__main__":
    main()
