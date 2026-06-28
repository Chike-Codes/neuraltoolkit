import requests
from pathlib import Path
from neuraltoolkit import CLI
from neuraltoolkit.datasets.management.data_resource import DatasetResource
from neuraltoolkit.datasets.management.paths import DATASETS_PATH
from neuraltoolkit.datasets.management.verify import verify_sha256

def download(resource:DatasetResource):
    CHUNK_SIZE = 8192

    response = requests.get(resource.url, stream=True)
    response.raise_for_status()

    response_size = int(response.headers.get("Content-Length"))
    progress = 1

    destination_path = DATASETS_PATH / resource.file_name

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {resource.name}...")
    with open(destination_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)

            # Display the progress bar in the terminal
            progress += CHUNK_SIZE
            frac = min(1, progress // response_size)
            CLI.progress_bar(
                frac,
                end_str=f"{frac * 100}%"
                )
    
    if not verify_sha256(destination_path, resource.sha256):
        raise RuntimeError("Dataset Verification Failed.")
    
    print("\nDownload Successful")
