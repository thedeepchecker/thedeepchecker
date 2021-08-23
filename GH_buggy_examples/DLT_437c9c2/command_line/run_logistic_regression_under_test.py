import tensorflow as tf

from tf_models import logistic_regression
import datasets
from tfcheck import DNNChecker
import interfaces
import data 
# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# Global configuration
flags.DEFINE_string('dataset', 'mnist', 'Which dataset to use. ["mnist", "cifar10"]')
flags.DEFINE_string('cifar_dir', '', 'Path to the cifar 10 dataset directory.')
flags.DEFINE_string('directory_name', 'lr/', 'Directory to store data relative to the algorithm.')
flags.DEFINE_string('loss_func', 'cross_entropy', 'Loss function. ["mean_squared" or "cross_entropy"]')
flags.DEFINE_integer('verbose', 0, 'Level of verbosity. 0 - silent, 1 - print accuracy.')
flags.DEFINE_string('optimizer', 'gradient_descent', '["gradient_descent", "ada_grad", "momentum"]')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_float('momentum', 0.5, 'Momentum parameter.')
flags.DEFINE_integer('n_iter', 10, 'Number of epochs.')
flags.DEFINE_integer('batch_size', 10, 'Size of each mini-batch.')

assert FLAGS.dataset in ['mnist', 'cifar10']
assert FLAGS.loss_func in ['cross_entropy', 'mean_squared']
assert FLAGS.optimizer in ['gradient_descent', 'ada_grad', 'momentum']

if __name__ == '__main__':

    if FLAGS.dataset == 'mnist':

        # ################# #
        #   MNIST Dataset   #
        # ################# #

        trX, trY, vlX, vlY, teX, teY = datasets.load_mnist_dataset(mode='supervised')
        trX, teX = trX.reshape(-1,784), teX.reshape(-1,784)
        print(trY.shape)
    elif FLAGS.dataset == 'cifar10':

        # ################### #
        #   Cifar10 Dataset   #
        # ################### #

        trX, trY, teX, teY = datasets.load_cifar10_dataset(FLAGS.cifar_dir, mode='supervised')
        
        # Validation set is the first half of the test set
        vlX = teX[:5000]
        vlY = teY[:5000]

    else:  # cannot be reached, just for completeness
        trX = None
        trY = None
        vlX = None
        vlY = None
        teX = None
        teY = None

    data_loader_under_test = data.DataLoaderFromArrays(trX, trY, shuffle=True, one_hot=False, normalization=False)
    test_data_loader = data.DataLoaderFromArrays(vlX, vlY, shuffle=True, one_hot=False, normalization=False)
    
    # Create the object
    l = logistic_regression.LogisticRegression(
        dataset=FLAGS.dataset, loss_func=FLAGS.loss_func,
        learning_rate=FLAGS.learning_rate,
        verbose=FLAGS.verbose)

    l._build_model(trX.shape[1], trY.shape[1])

    model_under_test =  interfaces.build_model_interface(problem_type='classification', 
                          test_mode=False,
                          features=l.input_data, 
                          targets=l.input_labels, 
                          train_op=l.train_step, 
                          loss=l.cost, 
                          reg_loss=None,
                          perf=l.accuracy, 
                          perf_dir='Up', 
                          outs_pre_act=l.logits, 
                          outs_post_act=l.model_output)
    data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
    checker = DNNChecker(name='DLT_437c9c2', data=data_under_test, model=model_under_test, buffer_scale=10)
    checker.run_full_checks()

#python ./command_line/run_logistic_regression_under_test.py --dataset mnist  --verbose 1 --learning_rate 1e-2 --optimizer gradient_descent