import argparse
from train import train

from test import test
import params

def main():

    parser = argparse.ArgumentParser()

    # parameters log
    parser.add_argument('--model_name', type=str, default="3dgan")
    parser.add_argument('--test', type=bool, default=False)
    parser.add_argument('--use_visdom', type=bool, default=False)
    parser.add_argument('--use_checkpoint', type=bool, default=False)
    parser.add_argument('--make_2D', type=bool, default=False)
    args = parser.parse_args()

    params.print_params()

    # run program
    if args.test == False:
        train(args)
    else:
        test(args)

if __name__ == '__main__':
    main()

    
