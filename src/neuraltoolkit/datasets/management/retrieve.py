from neuraltoolkit.datasets.management.data_resource import DatasetResource
from neuraltoolkit.datasets.management.downloader import download
from neuraltoolkit.datasets.management.paths import DATASETS_PATH
from neuraltoolkit.datasets.management.verify import verify_sha256


def retrieve_path(resource:DatasetResource):
    """Returns the path. It downloads it first if it doesn't exist"""
    path = DATASETS_PATH / resource.file_name
    if not path.exists() or not verify_sha256(path, resource.sha256):
        download(resource)

    return path