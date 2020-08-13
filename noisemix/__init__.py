import fasttext
import argparse

from noisemix import noise, formats
from noisemix import utils

NOISEMIX_SUFFIX = '.nmx'


def mix(dataset, versions=1, format=None):
    """Generates a noisy version of a dataset"""

    if format:
        format = eval('formats.' + args.format)  # string to cls
        format = format()  # cls to instance
    utils.rand_init()
    for line in dataset:
        # print(line)

        if format:
            line, data = format.before(line)

        for i in range(0, versions):
            line = mix_sentence(line);
            if format:
                line = format.after(line, data)
            yield line


def mix_sentence(sentence):
    words = []
    for word in sentence.split(' '):
        words.append(noise.randnoise(word))
    # noisemix.append_noise(words)
    sentence = (' ').join(words)
    return noise.sentence_noise(sentence)


def _read(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line


def _write(dataset, path):
    with open(path, 'w', encoding='utf-8') as f:
        for line in dataset:
            f.write(line)


def _print(dataset, max_lines=None):
    line_count = 0
    for line in dataset:
        print(line)
        line_count += 1
        if line_count == max_lines:
            return


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Generate a noisy version of dataset')
    parser.add_argument('path', metavar='path', type=str,
                        help='the paths to the input data file')
    parser.add_argument('-format', type=str, choices=[None, 'fastText'],
                        help='the format of the input data file', required=True)
    parser.add_argument('-versions', type=int, default=1,
                        help='how many versions to generate per line')
    parser.add_argument('--print', type=int, nargs='?', const=max,
                        help='print the output instead of writing to file, optional value for max lines')
    # parser.add_argument('-sentencePerturbations', nargs='?' ,action='append', default=['all'],
    #                     choices=['all', 'none','remove_space', 'flip_words'] )
    # parser.add_argument('-wordPerturbations', nargs='?' , action='append', default=['all'],
    #                     choices=['all', 'none', 'add_letter', 'repeat_letter', 'remove_letter',
    #                              'lowercase','remove_punct','word_swap','char_swap',
    #                              'flip_letters','typo_qwerty'] )
    # parser.add_argument('-perturbationFreqRepeat', nargs='?', action='append', default=[{'all', 1, 0.3}])
    # parser.add_argument('-language', type=str, default='eng', choices=['eng'])
    # parser.add_argument('-keyboardLayout', nargs='?', type=str, default='qwerty', choices=[ 'qwerty'])
    args = parser.parse_args()

    dataset = _read(args.path)

    classifier = fasttext.supervised(args.path , 'model')
    result = classifier.test('/Users/anooshik/Projects/CODE/Pycharm/NoiseMix/benchmarks/LangDetection/valid.txt')
    print('Init Precision:', result.precision)
    print('Init Recall:', result.recall)
    print('Init Number of examples:', result.nexamples)
    mixed_dataset = mix(dataset, versions=args.versions, format=format)
    if args.print:
        _print(mixed_dataset, args.print)
    else:
        _write(mixed_dataset, args.path + NOISEMIX_SUFFIX)
        classifier = fasttext.supervised(args.path + NOISEMIX_SUFFIX, 'model')
        result = classifier.test('/Users/anooshik/Projects/CODE/Pycharm/NoiseMix/benchmarks/LangDetection/valid.txt')
        print('Precision:', result.precision)
        print('Recall:', result.recall)
        print('Number of examples:', result.nexamples)
