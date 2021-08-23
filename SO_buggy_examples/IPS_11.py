import tensorflow as tf
from tfcheck import DNNChecker
import interfaces
import data 

tf.set_random_seed(20180130)

def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b, name='relu')
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        activation = tf.nn.relu(tf.matmul(input, w) + b, name='relu')
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", activation)
        return activation


def mnist_model(learning_rate, path):
    #tf.reset_default_graph()
    #sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

    conv1 = conv_layer(x_image, 1, 32, "conv1")
    conv_out = conv_layer(conv1, 32, 64, "conv2")

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])

    fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, name="fc1")
    logits = fc_layer(fc1, 1024, 10, name="fc2")
    probabilities = tf.nn.softmax(logits)

    with tf.name_scope("xent"):
        xent = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name="xent")
        tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0],-1)
    x_test = x_test.reshape(x_test.shape[0],-1)
    data_loader_under_test = data.DataLoaderFromArrays(x_train, y_train, shuffle=True, one_hot=True, normalization=True)
    test_data_loader = data.DataLoaderFromArrays(x_test, y_test, shuffle=True, one_hot=True, normalization=True)
    model_under_test =  interfaces.build_model_interface(problem_type='classification', 
                                                            test_mode=False,
                                                            features=x, 
                                                            targets=y, 
                                                            train_op=train_step, 
                                                            loss=xent, 
                                                            perf=accuracy, 
                                                            perf_dir='Up', 
                                                            outs_pre_act=logits, 
                                                            outs_post_act=probabilities, 
                                                            weights=['conv1/W:0', 'conv2/W:0','fc1/W:0', 'fc2/W:0'], 
                                                            biases=['conv1/B:0', 'conv2/B:0', 'fc1/B:0', 'fc2/B:0'], 
                                                            activations=['conv1/relu', 'conv2/relu', 'fc1/relu'],
                                                            test_activations=None)
    data_under_test = interfaces.build_data_interface(data_loader_under_test, test_data_loader, homogeneous=True)
    checker = DNNChecker(name='IPS_11', data=data_under_test, model=model_under_test, buffer_scale=10)
    checker.run_full_checks()

mnist_model(1e-3, path="/tmp/mnist_demo/10")
