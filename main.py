import argparse, yaml, pprint, os

from kan_classifiers.train.train import train

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config.yaml file to load',
    default='/app/kan_classifiers/kan_classifiers/configs/exp1.yaml'
)
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='the ID for the device(s) to use.'
)


def process_main(args):
    import logging
    from kan_classifiers.utils.logging import get_logger

    params = None
    with open(args.fname, 'r') as config_file:
        params = yaml.load(config_file, Loader=yaml.FullLoader)

    # LOGGING
    log_folder = params['logging']['folder']
    exp_name = params['experiment']['exp_name']

    log_exp_path = os.path.join(log_folder, exp_name)
    if not os.path.exists(log_exp_path):
        os.mkdir(log_exp_path)
    info_logger = get_logger(filename=os.path.join(log_exp_path, 'info_train.log'), force=True)
    info_logger.setLevel(logging.INFO)
    info_logger.info(f'Created log file at: {log_exp_path}')

    params_str = pprint.pformat(params, indent=4)

    info_logger.info(f'Parameters:\n{params_str}')

    info_logger.info(f"Running experiments {params['experiment']['exp_name']}")
    train(params)



if __name__ == '__main__':
    args = parser.parse_args()
    # single gpu only for now
    process_main(args)