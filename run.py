import sys
import argparse
from pathlib import Path
from preprocessor import get_dataset_preprocessor, DatasetPreprocessorNotFoundError
from analyzer import get_dataset_processor, DatasetNotFoundError


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
    parser = argparse.ArgumentParser(usage='run.py [-h] preprocess --dataset {aesw,lang8,fce,jfleg} '
                                           '--dataset_path DATASET_PATH')
    parser.add_argument('--dataset', choices=['aesw', 'lang8', 'fce', 'jfleg'], type=str, required=True, help='dataset name')
    parser.add_argument('--dataset_path', type=str, required=True, help='a path to the directory with specified dataset')
    args = parser.parse_args(sys.argv[2:])
    preprocess_dataset(args.dataset, args.dataset_path)


def process_dataset(dataset_name: str):
    try:
        dataset_processor = get_dataset_processor(dataset_name)
    except DatasetNotFoundError as err:
        print(err.message, file=sys.stderr)
        return
    dataset_processor.compute_metrics()


def parse_analyze_command():
    parser = argparse.ArgumentParser(usage='run.py [-h] analyze --dataset {aesw,lang8,fce,jfleg}')
    parser.add_argument('--dataset', choices=['aesw', 'lang8', 'fce', 'jfleg'], type=str, required=True, help='dataset name')
    args = parser.parse_args(sys.argv[2:])
    process_dataset(args.dataset)


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
            '    --dataset {aesw,lang8,fce,jfleg}'
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
