import argparse
from utils import seed_everything, get_args_parser_working_with_model

def main(args):
   print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Gaze Transformer Test', parents=[get_args_parser_working_with_model()])
    args = parser.parse_args()
    main(args)
