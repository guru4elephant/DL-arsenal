import argparse

def parse_args():
        parser = argparse.ArgumentParser(description="Text classification on IMDB")
        parser.add_argument(
                '--train_data_path',
                type=str,
                default='./train_data',
                help="The path of training dataset")
        parser.add_argument(
                '--test_data_path',
                type=str,
                default='./test_data',
                help="The path of testing dataset")
        parser.add_argument(
                '--batch_size',
                type=int,
                default=128,
                help="The size of mini-batch (default:128)")
        parser.add_argument(
                '--embedding_size',
                type=int,
                default=10,
                help="The size for embedding layer (default:10)")
        parser.add_argument(
                '--num_passes',
                type=int,
                default=10,
                help="The number of passes to train (default: 10)")
        parser.add_argument(
                '--model_output_dir',
                type=str,
                default='models',
                help='The path for model to store (default: models)')
        parser.add_argument(
                '--dataset_mode',
                type=str,
                default='QueueDataset',
                help= 'QueueDataset or InMemoryDataset (default: QueueDataset)')
        parser.add_argument(
                '--thread',
                type=int,
                default=10,
                help='thread number for dataset parsing and training')
        parser.add_argument(
                '--text_encoder',
                type=str,
                default='bow',
                help='bow/cnn/gru/lstm (default: bow)')
        return parser.parse_args()
