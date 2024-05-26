import argparse, yaml, pprint, os

from kan_classifiers.train import train

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config.yaml file to load',
    default='exp1.yaml'
)
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='the ID for the device(s) to use.'
)


def process_main(args):
    import logging
    from kan_classifiers.utils.logging import get_logger

    logger = get_logger(force=True)
    logger.setLevel(logging.INFO)

    logger.info(f'Load parameters from {args.fname}')

    params = None
    with open(args.fname, 'r') as config_file:
        params = yaml.load(config_file, Loader=yaml.FullLoader)
        logger.info('parameters loaded')

    pprint.PrettyPrinter(indent=4).pprint(params)
    dump = os.path.join(params['logging']['folder'], f'{params['experiment']['exp_name']}.log')
    with open(dump, 'w') as f:
        yaml.dump(params, f)

    logger.info(f'Running experiments {params['experiment']['exp_name']}')
    train(params)



if __name__ == '__main__':
    args = parser.parse_args()
    # single gpu only for now
    process_main(args)