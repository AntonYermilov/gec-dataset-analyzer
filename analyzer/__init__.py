from .dataset import Dataset, DatasetNotFoundError
from .process import DatasetProcessor


def get_dataset_processor(dataset_name: str):
    return DatasetProcessor(Dataset(dataset_name))
