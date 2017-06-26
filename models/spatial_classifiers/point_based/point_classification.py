# Compatibility imports
from __future__ import with_statement, print_function, division, absolute_import
import six
from six.moves import range
from six.moves import zip

import numpy as np
import sklearn.preprocessing

import subprocess
import concurrent.futures as futures

from . import make_samples_linear




def get_acc(preds_tf, labels_tf):
    assert False, "test this.. really not sure"
    return tf.reduce_mean(tf.equal(tf.argmax(preds_tf, axis=1), labels_tf), axis=0)


def train():
    inputs_tf, labels_tf = placeholder_inputs(batch_size, num_point, depth)
    preds_tf, _ = get_model(inputs_tf, is_training_tf)
    loss_tf = get_loss(preds_tf, labels_tf)
    softmax_preds_tf = tf.nn.softmax(preds_tf)
    acc_tf = get_acc(preds_tf, labels_tf)
    step_tf = tf.Variable(0)
    training_tf = tf.train.AdamOptimizer().minimize(loss_tf, global_step=step_tf)
    init_tf = tf.initialize_all_variables()
    tf.summary.scalar('accuracy', acc_tf)


    
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
    test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

    with tf.Session() as tf_session:
        tf_session.run([init_tf])

        for epoch in epochs:
            for inputs_np, labels_np in epoch_tr:                
                    feed_dict = {
                        inputfs_tf: inputs_np,
                        labels_tf:  labels_np,
                        is_training_tf: True,

                    }
                    _, softmax_pred_np, loss_np, acc_np = tf_session.run([training_tf, softmax_preds_tf, loss_tf, acc_tf], feed_dict=feed_dict)


            for inputs_np, labels_np in epoch_va:
                for inputs_np, labels_np in epoch_va:                
                    feed_dict = {
                        inputfs_tf: inputs_np,
                        labels_tf:  labels_np,
                        is_training_tf: false,
                        }
                    softmax_pred_np, loss_np, acc_np = tf_session.run([softmax_preds_tf, loss_tf, acc_tf], feed_dict=feed_dict)


def experiment(x, y):
    linear_x = [None, None, None]

    for i in xrange(3):
        linear_x[i] = make_samples_linear(x[i])

    scaler = sklearn.preprocessing.StandardScaler()
    linear_x[0] = scaler.fit_transform(linear_x[0])
    linear_x[1] = scaler.transform(linear_x[1])
    linear_x[2] = scaler.transform(linear_x[2])

    assert len(linear_x) == 3
    assert len(y) == 3

    training_x = linear_x[0]
    valid_x = linear_x[1]
    test_x = linear_x[2]

    training_y = y[0]
    valid_y = y[1]
    test_y = y[2]

    

    def chain(cl, training_x, training_y, valid_x):
        """
        """
        cl.fit(training_x, training_y)
        preds = cl.predict(valid_x)
        return cl, preds, cl.score(training_x, training_y), cl.score(valid_x, valid_y), np.mean(preds)


    c_const = 10.
    arg_combinations = []
    classifier_futures = []
    c_exp = 0.01
    

    with futures.ThreadPoolExecutor(max_workers=int(subprocess.check_output("nproc")) - 1) as executor:
        for max_iter in range(29, 60, 2):
            cl = SVC(C=c_const ** c_exp,
                    kernel="linear",
                    max_iter=max_iter * 100
                    )
            classifier_futures.append(executor.submit(chain, cl, training_x, training_y, valid_x))
                    
                    
        print("--")
        print("Doing linear classification")
        for classifier_future in classifier_futures:
            cl, preds, tr_score, va_score, mean_preds = classifier_future.result()
            print("\t- classif obj:      {}".format(cl))
            print("\t- Training score:   {}".format(tr_score))
            print("\t- Valid score:      {}".format(va_score))
            print("\t- valid avg:        {}".format(mean_preds))
            print("\t--")

