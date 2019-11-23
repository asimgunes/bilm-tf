
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab_file = os.path.join(args.folder, 'vocabulary.txt')
    vocab = load_vocab(vocab_file, 50)

    # define the options
    batch_size = 128  # batch size for each GPU
    n_gpus = args.gpu

    # number of tokens in training data (this for 1B Word Benchmark)
    #n_train_tokens = 768648884
    n_train_tokens = args.tokens

    options = {
     'bidirectional': True,

     'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [[1, 32],
       [2, 32],
       [3, 64],
       [4, 128],
       [5, 256],
       [6, 512],
       [7, 1024]],
      'max_characters_per_token': 50,
      'n_characters': 261,
      'n_highway': 2},
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 512,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': args.epoch,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
     'n_negative_samples_batch': 8192,
    }

    prefix = os.path.join(args.folder, 'corpus', '*')
    data = BidirectionalLMDataset(prefix, vocab, test=False, shuffle_on_load=True)

    tf_save_dir = os.path.join(args.folder, args.checkpoint)
    tf_log_dir = tf_save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--save_dir', help='Location of checkpoint files')
    #parser.add_argument('--vocab_file', help='Vocabulary file')
    #parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('folder', help='Training folder for biLM')
    parser.add_argument('--tokens', help='Token count to train', type=int)
    parser.add_argument('--checkpoint', help='Checkpoint folder for training data', default='checkpoint')
    parser.add_argument('--gpu', help='Number of GPU to use in the training', type=int, default=1)
    parser.add_argument('--epoch', help='Number epoch to traing', type=int, default=10)

    args = parser.parse_args()
    main(args)

