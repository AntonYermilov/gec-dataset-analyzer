from .aesw import AESWPreprocessor
from .lang8 import Lang8Preprocessor
from .fce import FCEPreprocessor
from .jfleg import JFLEGPreprocessor


class DatasetPreprocessorNotFoundError(ValueError):
    def __init__(self, preprocessor_name: str):
        super().__init__()
        self.message = f'Dataset preprocessor with name {preprocessor_name} was not found.'


def get_dataset_preprocessor(preprocessor_name: str):
    preprocessors = {
        'aesw': AESWPreprocessor,
        'lang8': Lang8Preprocessor,
        'fce': FCEPreprocessor,
        'jfleg': JFLEGPreprocessor
    }
    if preprocessor_name not in preprocessors:
        raise DatasetPreprocessorNotFoundError(preprocessor_name=preprocessor_name)
    return preprocessors[preprocessor_name]
