from typing import Optional
from .dataset import Dataset, DatasetNotFoundError
from .process import DatasetProcessor


def get_dataset_processor(dataset_name: str, only_edited: bool, sample_rate: float):
    return DatasetProcessor(Dataset(dataset_name, only_edited, sample_rate))
