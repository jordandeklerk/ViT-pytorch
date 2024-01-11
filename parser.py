import argparse

parser = argparse.ArgumentParser(description='PyTorch Transformer Vision Model')

parser.add_argument('--DATA_DIR', type=str, default='./data',
                    help='Data directory')
parser.add_argument('--BATCH_SIZE', type=int, default=128,
                    help='batch size')
parser.add_argument('--IMAGE_SIZE', type=int, default=32,
                    help='image size')
parser.add_argument('--EPOCHS', type=int, default=200,
                    help='number of epochs')
parser.add_argument('--LEARNING_RATE', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--WEIGHT_DECAY', type=float, default=1e-1,
                    help='weight decay')
parser.add_argument('--CYCLE_MULT', type=int, default = 1.0,
                    help='cycle mult for Cosine LR schedule')
parser.add_argument('--MIN_LR', type=float, default=0.00001,
                    help='Minimum learning rate for Cosine LR schedule')
parser.add_argument('--WARM_UP', type=int, default=10,
                    help='Warm up steps for Cosine LR schedule')
parser.add_argument('--GAMMA', type=float, default=1.0,
                    help='Gamma value for Cosine LR schedule')
parser.add_argument('--ALPHA', type=float, default=1.0,
                    help='Alpha for cutmix')
parser.add_argument('--LABEL_SMOOTHING', type=float, default=0.1,
                    help='Label smoothing for optimizer')
parser.add_argument('--NUM_CLASSES', type=int, default=10,
                    help='Number of classes in the data')
parser.add_argument('--NUM_WORKERS', type=int, default=8,
                    help='Number of workers for GPU')
parser.add_argument('--CHANNELS', type=int, default=256,
                    help='Embedding dimension')
parser.add_argument('--HEAD_CHANNELS', type=int, default=32,
                    help='Head embedding dimension')
parser.add_argument('--NUM_BLOCKS', type=int, default=8,
                    help='Number of transformer blocks')
parser.add_argument('--PATCH_SIZE', type=int, default=2,
                    help='Patch size for patch embedding')
parser.add_argument('--EMB_P_DROP', type=float, default=0.,
                    help='Embedding dropout probability')
parser.add_argument('--TRANS_P_DROP', type=float, default=0.,
                    help='Transformer block droout probability')
parser.add_argument('--HEAD_P_DROP', type=float, default=0.3,
                    help='Head dropout probability')

args, unknown = parser.parse_known_args()