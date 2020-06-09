import sys
import argparse
from pathlib import Path
from typing import Optional
from preprocessor import get_dataset_preprocessor, DatasetPreprocessorNotFoundError
from metric_analyzer import get_dataset_processor, DatasetNotFoundError


def preprocess_dataset(dataset_name: str, dataset_path: str):
    try:
        dataset_path = Path(dataset_path)
        dataset_preprocessor = get_dataset_preprocessor(dataset_name)(dataset_path)
    except DatasetPreprocessorNotFoundError as err:
        print(err.message, file=sys.stderr)
        return

    try:
        dataset_preprocessor.preprocess()
    except Exception as err:
        print(f'Can\'t process dataset {dataset_name}, possibly invalid path to the dataset was provided.\n'
              f'Please check the description of the specified dataset preprocessor.')
        print(f'Error: {err}', file=sys.stderr)
        return


def parse_preprocess_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['aesw', 'lang8', 'fce', 'jfleg'], type=str, required=True,
                        help='dataset name')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='a path to the directory with specified dataset')
    args = parser.parse_args(sys.argv[2:])
    preprocess_dataset(args.dataset, args.dataset_path)


def process_dataset(dataset_name: str, only_edited: bool, sample_rate: float, extract_edits: bool):
    try:
        dataset_processor = get_dataset_processor(dataset_name, only_edited, sample_rate)
    except DatasetNotFoundError as err:
        print(err.message, file=sys.stderr)
        return
    if extract_edits:
        dataset_processor.extract_edits()
    else:
        dataset_processor.compute_metrics()


def parse_analyze_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['aesw', 'lang8', 'fce', 'jfleg', 'papeeria'], type=str, required=True,
                        help='dataset name')
    parser.add_argument('--only-edited', action='store_true',
                        help='use only pairs of sentences with edits')
    parser.add_argument('--sample-rate', type=float, default=1.0,
                        help='use only pairs of sentences with edits')
    parser.add_argument('--extract-edits', action='store_true',
                        help='extract sentences with substitutions')
    args = parser.parse_args(sys.argv[2:])
    process_dataset(args.dataset, args.only_edited, args.sample_rate, args.extract_edits)


def get_action_type():
    if len(sys.argv) == 1:
        raise NameError()
    action_type = sys.argv[1]
    if action_type not in {'preprocess', 'analyze'}:
        raise NameError()
    return action_type


def main():
    try:
        action_type = get_action_type()
    except NameError:
        print(
            'usage: run.py [-h] {preprocess,analyze}\n'
            '  preprocess\n'
            '    --dataset {aesw,lang8,fce,jfleg}\n'
            '    --dataset_path DATASET_PATH\n'
            '  analyze\n'
            '    --dataset {aesw,lang8,fce,jfleg,papeeria}\n'
            '    [--only-edited]\n'
            '    [--sample-rate RATE]\n'
            '    [--extract-subst]\n'
        )
        return

    actions = {
        'preprocess': parse_preprocess_command,
        'analyze': parse_analyze_command
    }
    actions[action_type]()


def install_dependencies():
    try:
        from nltk import word_tokenize
        word_tokenize('Hello world!')
    except:
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        try:
            import nltk
            print('Installing nltk.punkt')
            nltk.download('punkt', raise_on_error=True)
        except:
            print('Unable to download nltk.punkt, check your internet connection', file=sys.stderr)
            return False
    return True


if __name__ == '__main__':
    if install_dependencies():
        main()
