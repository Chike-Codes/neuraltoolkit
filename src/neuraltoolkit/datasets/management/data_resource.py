from dataclasses import dataclass


@dataclass
class DatasetResource:
    name:str
    url:str
    sha256:str
    file_name:str
