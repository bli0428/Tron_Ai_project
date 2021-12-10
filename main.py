import logging
from trainer import Trainer
from agent import Net
import argparse

log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(prog="main", usage="%(prog)s --load_model <path/to/model>")
    parser.add_argument(
        "-load_model", type=str,
    )
    args = parser.parse_args()

    maps = get_maps()

    log.info('Loading Network...')
    net = Net()

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_model)
        net.load(args.load_model)
    else:
        log.warning('Failed to load a checkpoint! Starting from scratch...')

    log.info('Loading Trainer...')
    # TODO: Add all Trainer Parameters
    trainer = Trainer(maps, net)

    if args.load_model:
        log.info("Loading 'train_examples' from file...")
        trainer.load_train_examples()

    log.info('Starting the learning process...')
    trainer.learn()

def get_maps():
    map_paths = ['center_block','diagonal_blocks','divider','empty_room','hunger_games','joust','small_room']
    return map_paths

if __name__ == "__main__":
    main()