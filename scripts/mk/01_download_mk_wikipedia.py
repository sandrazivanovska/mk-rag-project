from pathlib import Path
import requests
from tqdm import tqdm

URL = "https://dumps.wikimedia.org/mkwiki/latest/mkwiki-latest-pages-articles.xml.bz2"

OUTPUT_PATH = Path("data/raw/mk_wikipedia/mkwiki-latest-pages-articles.xml.bz2")


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as file, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc="Downloading Macedonian Wikipedia",
    ) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
                progress.update(len(chunk))

    print(f"Downloaded to: {output_path}")


if __name__ == "__main__":
    download_file(URL, OUTPUT_PATH)