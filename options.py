from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer('h_dim', default=32,
                     help='Hidden dim in various models.')

flags.DEFINE_integer('rnn_dim', default=256,
                     help='RNN hidden dim.')

flags.DEFINE_integer('rnn_n_layers', default=2,
                     help='Number of layers for RNNs.')

flags.DEFINE_float('rnn_drop', default=0.1,
                   help='Dropout rate in RNNs.')

flags.DEFINE_integer('n_latent', default=24,
                     help='Latent dimension for vaes.')

flags.DEFINE_integer('n_batch', default=128,
                     help='Minibatch size to train.')

flags.DEFINE_integer('visualize_every', default=10,
                     help='Frequency of visualization.')

flags.DEFINE_integer('n_iter', default=200000,
                     help='Number of iteration to train. Might not be used if '
                          'n_epoch is used.')

flags.DEFINE_integer('n_epoch', default=50,
                     help='Number of epochs to train. Might not be used if '
                          'n_iter is used.')

flags.DEFINE_integer('n_workers', default=4,
                     help='Sets num workers for data loaders.')

flags.DEFINE_integer('seed', default=0,
                     help='Sets global seed.')

flags.DEFINE_string("vis_root", default='vis',
                    help='root folder for visualization and logs.')

flags.DEFINE_float('decay', default=0.99,
                   help='set learning rate value for optimizers')

flags.DEFINE_float('lr', default=1e-3,
                   help='Set learning rate for optimizers.')

flags.DEFINE_bool("debug", default=False,
                  help='Enables debug mode.')

flags.DEFINE_bool('highdrop', default=False,
                  help='Enables high dropout to encourage copy.')

flags.DEFINE_bool('highdroptest', default=False,
                  help='Applies high dropout in test as well.')

flags.DEFINE_float("highdropvalue", default=0.,
                   help='High dropout value to encourage copying.')

flags.DEFINE_bool('copy', default=False,
                  help='Enable copy in seq2seq models')

flags.DEFINE_string('model_path', default='',
                    help="Model path to load a pretrained model")

flags.DEFINE_bool('extract_codes', default=False,
                  help='Extract VQVAE codes for training and test set given a '
                       'pretrained vae')

flags.DEFINE_bool('filter_model', default=False,
                  help='To run filter model experiments.')

flags.DEFINE_bool('test', default=False,
                  help='Only runs evaluations.')


flags.DEFINE_string('tensorboard', default=None,
                    help='Use tensorboard for logging losses.')

flags.DEFINE_bool('kl_anneal', default=False,
                  help='Enables kl annealing.')

flags.DEFINE_integer('decoder_reset', default=-1,
                     help='Enables decoder reset for vae to prevent posterior collapse.')

flags.DEFINE_string("resume", default='',
                    help='Path to the main model to resume training')
