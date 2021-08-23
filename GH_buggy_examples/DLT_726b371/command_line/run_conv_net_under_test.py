import tensorflow as tf
import numpy as np
from tfcheck import DNNChecker
import interfaces
import data 
from models.convolutional_models import conv_net
import utilities
import datasets
# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10", "custom"]')
flags.DEFINE_string('original_shape', '28,28,1', 'Original shape of the images in the dataset.')
flags.DEFINE_string('train_dataset', '', 'Path to train set .npy file.')
flags.DEFINE_string('train_labels', '', 'Path to train labels .npy file.')
flags.DEFINE_string('valid_dataset', '', 'Path to valid set .npy file.')
flags.DEFINE_string('valid_labels', '', 'Path to valid labels .npy file.')
flags.DEFINE_string('test_dataset', '', 'Path to test set .npy file.')
flags.DEFINE_string('test_labels', '', 'Path to test labels .npy file.')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_string('model_name', 'convnet', 'Model name.')
flags.DEFINE_string('main_dir', 'convnet/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_integer('verbose', 0, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_boolean('restore_previous_model', False, 'If true, restore previous model corresponding to model name.')
flags.DEFINE_integer('seed', -1, 'Seed for the random generators (>= 0). Useful for testing hyperparameters.')


# Convolutional Net parameters
flags.DEFINE_string('layers', '', 'String representing the architecture of the network.')
flags.DEFINE_string('loss_func', 'cross_entropy', 'Loss function. ["mean_squared" or "cross_entropy"]')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 10, 'Size of each mini-batch.')
flags.DEFINE_string('opt', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum", "adam"]')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')
flags.DEFINE_float('dropout', 1, 'Dropout parameter.')

assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
assert FLAGS.opt in ['gradient_descent', 'ada_grad', 'momentum', 'adam']

if __name__ == '__main__':

    utilities.random_seed_np_tf(FLAGS.seed)

    if FLAGS.dataset == 'mnist':

        # ################# #
        #   MNIST Dataset   #
        # ################# #

        trX, trY, vlX, vlY, teX, teY = datasets.load_mnist_dataset(mode='supervised')

    elif FLAGS.dataset == 'cifar10':

        # ################### #
        #   Cifar10 Dataset   #
        # ################### #

        trX, trY, teX, teY = datasets.load_cifar10_dataset(FLAGS.cifar_dir, mode='supervised')
        vlX = teX[:5000]  # Validation set is the first half of the test set
        vlY = teY[:5000]

    elif FLAGS.dataset == 'custom':

        # ################## #
        #   Custom Dataset   #
        # ################## #

        def load_from_np(dataset_path):
            if dataset_path != '':
                return np.load(dataset_path)
            else:
                return None


        trX, trY = load_from_np(FLAGS.train_dataset), load_from_np(FLAGS.train_labels)
        vlX, vlY = load_from_np(FLAGS.valid_dataset), load_from_np(FLAGS.valid_labels)
        teX, teY = load_from_np(FLAGS.test_dataset), load_from_np(FLAGS.test_labels)

    else:
        trX, trY, vlX, vlY, teX, teY = None, None, None, None, None, None

    data_loader_under_test = data.DataLoaderFromArrays(trX, trY, shuffle=True, one_hot=False, normalization=False)
    test_data_loader = data.DataLoaderFromArrays(vlX, vlY, shuffle=True, one_hot=False, normalization=False)

    # Create the model object
    convnet = conv_net.ConvolutionalNetwork(
        layers=FLAGS.layers, model_name=FLAGS.model_name, main_dir=FLAGS.main_dir, loss_func=FLAGS.loss_func,
        num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size, dataset=FLAGS.dataset, opt=FLAGS.opt,
        learning_rate=FLAGS.learning_rate, momentum=FLAGS.momentum, dropout=FLAGS.dropout, verbose=FLAGS.verbose
    )

    # Model training
    print('Start Convolutional Network training...')
    convnet.build_model(trX.shape[1], trY.shape[1], [int(i) for i in FLAGS.original_shape.split(',')])

        
    model_under_test =  interfaces.build_model_interface(problem_type='classification', 
                          test_mode=False,
                          features=convnet.input_data, 
                          targets=convnet.input_labels, 
                          train_op=convnet.train_step, 
                          loss=convnet.cost, 
                          reg_loss=None,
                          perf=convnet.accuracy, 
                          perf_dir='Up', 
                          activations=['relu','relu_1','relu_2'],
                          outs_pre_act=convnet.logits, 
                          outs_post_act=convnet.softmax_out,
                          test_extra_feed_dict={convnet.keep_prob:1.0}, 
                          train_extra_feed_dict={convnet.keep_prob:FLAGS.dropout})
    data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
    checker = DNNChecker(name='DLT_726b371', data=data_under_test, model=model_under_test, buffer_scale=10)
    checker.run_full_checks()

    #python ./command_line/run_conv_net_under_test.py --dataset mnist --main_dir convnet_models --model_name convnet --layers conv2d-5-5-32-1,maxpool-2,conv2d-5-5-64-1,maxpool-2,full-1024,softmax --batch_size 50 --verbose 1 --learning_rate 1e-4 --opt adam --dropout 0.5