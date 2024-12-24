import os
import argparse


def do_replacement(checked_txt, tgt_txt):
    '''
    This function replaces the sentences in tgt_txt with the lines in checked_txt,
    retain or merge the timestamps in front of the sentences,
    and saves the original tgt_txt as tgt_txt_old.txt.

    Pay attention:
    The count of the lines in checked_txt must smaller than or equal to the count of the lines in tgt_txt,
    otherwise, the script can't how to strip the timestamps.

    checked_txt may contain the following:
    The words before check.
    some words

    tgt_txt may contain the following:
    7.92-9.74: The words bafore cheek.
    9.74-11.74: some words

    eg.
    7.92-9.74: The words bafore cheek. -> 7.92-9.74: The words before check.
    or:
    7.92-9.74: The words bafore cheek
    9.74-11.74: some words
    -> 7.92-11.74: The words before check, some words
    '''
    with open(checked_txt, 'r') as f:
        checked_txt = f.read()
    with open(tgt_txt, 'r') as f:
        tgt_txt = f.read()

    checked_lines = checked_txt.split('\n')
    tgt_lines = tgt_txt.split('\n')

    assert len(checked_lines) <= len(tgt_lines), \
        'The count of the lines in checked_txt must smaller than or equal to the count of the lines in tgt_txt.'

    with open(tgt_txt.replace('.txt', '_old.txt'), 'w') as f:
        f.write(tgt_txt)

    with open(tgt_txt, 'w') as f:
        for i in range(len(checked_lines)):
            if checked_lines[i] != tgt_lines[i]:
                f.write(checked_lines[i] + '\n')
            else:
                f.write(tgt_lines[i] + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-checked_txt", type=str, required=True)
    parser.add_argument("-tgt_txt", type=str, required=True)
    args = parser.parse_args()
    do_replacement(args.checked_txt, args.tgt_txt)


if __name__ == '__main__':
    main()
