import pandas as pd
import numpy as np
import time
from Logger import log
from tensorflow.keras import backend as K
import tensorflow as tf


def apply_ema(predictions, alpha=0.3):
    smoothed = np.zeros_like(predictions)
    smoothed[0] = predictions[0]

    for t in range(1, len(predictions)):
        smoothed[t] = alpha * predictions[t] + (1 - alpha) * smoothed[t-1]

    return smoothed

def dict_to_one(dp_dict={}):

    """ Input a dictionary, return a dictionary that all items are
    set to one, use for disable dropout, drop-connect layer and so on.
    输入一个字典，返回一个所有项都设置为 1 的字典，
    用于禁用 dropout、drop-connect 等操作。
    Parameters
    ----------
    dp_dict : dictionary keeping probabilities date
    """
    return {x: 1 for x in dp_dict}


def modelsaver(network, path, epoch_identifier=None):

    if epoch_identifier:
        ifile = path + '_' + str(epoch_identifier)
    else:
        ifile = path

    network.save(ifile + '.h5',include_optimizer=True)  # save_format="tf"

    # network.save('testoptim',include_optimizer=True)  # save_format="tf"
    network.save_weights(ifile + '_weights' + '.h5')


def customfit(sess,
              network,
              cost,
              train_op,
              train_provider,
              x,   #样本
              y_,  #样本实际标签
              acc=None,  #acc传来的值也是NONE
              n_epoch=50, #传来100，50被覆盖
              print_freq=1,
              val_provider=None,
              save_model=-1,
              tra_kwag=None,
              val_kwag=None,
              save_path=None,
              epoch_identifier=None,
              earlystopping=True,
              min_epoch=5,
              patience=20):  # 原 5,提前停止
    """
        Training a given network model by the given cost function, dataset, n_epoch etc.

        Parameters
        ----------
        sess : TensorFlow session
            sess = tf.InteractiveSession()
        network : a TensorLayer layer
            the network will be trained
        train_op : a TensorFlow optimizer
            like tf.train.AdamOptimizer
        x : placeholder
            for inputs
        y_ : placeholder
            for targets
        acc : the TensorFlow expression of accuracy (or other metric) or None
            if None, would not display the metric
        batch_size : int
            batch size for training and evaluating
        n_epoch : int
            the number of training epochs
        print_freq : int
            display the training information every ``print_freq`` epochs
        X_val : numpy array or None
            the input of validation data
        y_val : numpy array or None
            the target of validation data
        eval_train : boolen
            if X_val and y_val are not None, it refects whether to evaluate the training data
    """
    # parameters for earlystopping
    best_valid = np.inf
    best_valid_acc = np.inf
    best_valid_epoch = min_epoch

    # Training info
    total_train_loss = []
    total_val_loss = []
    single_step_train_loss = []
    single_step_val_loss = []


    log("Start training the network ...")
    start_time_begin = time.time()
    for epoch in range(n_epoch):
        start_time = time.time()
        loss_ep = 0
        n_step = 0
        log("------------------------- Epoch %d of %d --------------------------" % (epoch + 1, n_epoch))

        for batch in train_provider.feed_chunk():  #yeild会一批一批返回数据，直到返回所有数据

            X_train_a, y_train_a = batch
            X_train_a = K.cast_to_floatx(X_train_a)  #?这是啥
            y_train_a = K.cast_to_floatx(y_train_a)
            
            feed_dict = {x: X_train_a, y_: y_train_a}
            #feed_dict.update(network.all_drop)  # enable noise layers
            loss, _ = sess.run([cost, train_op], feed_dict=feed_dict)
            loss_ep += loss
            n_step += 1
            #print("    batch {0:d}".format(n_step))
            #log(tf.trainable_variables())

            """
            for v in tf.trainable_variables():
                if v.name == 'conv2d_1/kernel:0':
                    value = sess.run(v)
                    print(value)
                    break
            """

            #for k, v in zip(variables_names, values):
            #   print(k, v)
        loss_ep = loss_ep / n_step  #每一epoch的loss
        log('第%d次epoch的训练loss_ep: %f' % (epoch + 1,loss_ep))

        if epoch >= 0 or (epoch + 1) % print_freq == 0:
            # evaluate the val error at each epoch.
            if val_provider is not None:
                log("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time)) #epoch从0开始，要+1
                log("Validation...")
                train_loss, train_acc, n_batch_train = 0, 0, 0 #---------------------------------------------
                for batch in train_provider.feed_chunk():   #这里对所有数据又训练了一次，为什么？？
                    X_train_a, y_train_a = batch
                    #dp_dict = dict_to_one(network.all_drop)  # disable noise layers
                    feed_dict = {x: X_train_a, y_: y_train_a}
                    #feed_dict.update(dp_dict)
                    if acc is not None:
                        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                        train_acc += ac
                    else:
                        err = sess.run(cost, feed_dict=feed_dict)
                    train_loss += err
                    n_batch_train += 1
                    single_step_train_loss.append(err)
                total_train_loss.append(train_loss/n_batch_train)
                log("   train loss/n_batch_train: %f" % (train_loss / n_batch_train))
                log("   train loss: %f, n_batch_train: %d" % (train_loss, n_batch_train))

                if acc is not None:
                    log("   train acc: %f" % (train_acc / n_batch_train))

                val_loss, val_acc, n_batch_val = 0, 0, 0 #------------------------------------------------

                for batch in val_provider.feed_chunk():  #进行验证
                    X_val_a, y_val_a = batch
                    # dp_dict = dict_to_one(network.all_drop)  # disable noise layers
                    feed_dict = {x: X_val_a, y_: y_val_a}
                    # feed_dict.update(dp_dict)
                    if acc is not None:
                        err, ac = sess.run([cost, acc], feed_dict=feed_dict)  # acc是none，这里有什么意义？？
                        val_acc += ac
                    else:
                        err = sess.run(cost, feed_dict=feed_dict)
                    val_loss += err
                    n_batch_val += 1
                    single_step_val_loss.append(err)
                log("    val loss: %f" % (val_loss / n_batch_val))
                total_val_loss.append(val_loss/n_batch_val)
                if acc is not None:
                    log("   val acc: %f" % (val_acc / n_batch_val))
            else:
                log('no validation')
                log("Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep))

        if earlystopping:
            if epoch >= min_epoch:
                log("Evaluate earlystopping parameters...")
                current_valid = val_loss / n_batch_val
                current_valid_acc = val_acc / n_batch_val
                current_epoch = epoch
                current_train_loss = train_loss / n_batch_train
                current_train_acc = train_acc / n_batch_train
                log('    Current valid loss was {:.6f}, acc was {:.6f}, '
                    'train loss was {:.6f}, acc was {:.6f} at epoch {}.'
                    .format(current_valid, current_valid_acc, current_train_loss, current_train_acc, current_epoch+1))
                if current_valid < best_valid:
                    best_valid = current_valid
                    best_valid_acc = current_valid_acc
                    best_valid_epoch = current_epoch

                    # save the model parameters
                    modelsaver(network=network, path=save_path, epoch_identifier=None)
                    log('Best valid loss was {:.6f} and acc {:.6f} at epoch {}.'.format(
                          best_valid, best_valid_acc, best_valid_epoch+1))
                    print('best_valid的模型已保存！',best_valid)
                elif best_valid_epoch + patience < current_epoch:
                    log('Early stopping.')
                    log('Best valid loss was {:.6f} and acc {:.6f} at epoch {}.'.format(
                          best_valid, best_valid_acc, best_valid_epoch+1))
                    print('best_valid的模型已保存！',best_valid)
                    break

        else:
            current_val_loss = val_loss / n_batch_val
            current_val_acc = val_acc / n_batch_val
            current_epoch = epoch
            current_train_loss = train_loss / n_batch_train
            current_train_acc = train_acc / n_batch_train
            log('    Current valid loss was {:.8f}, acc was {:.6f}, train loss was {:.8f}, acc was {:.6f} at epoch {}.'
                .format(current_val_loss, current_val_acc, current_train_loss, current_train_acc, current_epoch+1))
            
#             print('best_valid的值为：',best_valid)
            
             #-----------------保存最好的模型--------------------------------------------------
            if current_val_loss < best_valid:  # best_valid在前面初始化为inf
                    best_valid = current_val_loss
                    best_valid_acc = current_val_acc
                    best_valid_epoch = current_epoch

                    # save the model parameters
                    modelsaver(network=network, path=save_path, epoch_identifier=None)
                    log('Best valid loss was {:.8f} and acc {:.6f} at epoch {}.'.format(
                          best_valid, best_valid_acc, best_valid_epoch+1))
                    print('best_valid的模型已保存！',best_valid)
            #---------------------------------------------------------------------------------
                    
            #log(save_model > 0, epoch % save_model == 0, epoch/save_model > 0)
            if save_model > 0 and epoch % save_model == 0:
                if epoch_identifier:
                    modelsaver(network=network, path=save_path, epoch_identifier=epoch+1)
                else:
                    modelsaver(network=network, path=save_path, epoch_identifier=None)
        #for epoch in range(n_epoch)结束
    # if not earlystopping:
    #     if save_model == -1:
    #         modelsaver(network=network, path=save_path, epoch_identifier=None)

    log("Total training time: %fs" % (time.time() - start_time_begin))
    return total_train_loss, total_val_loss, single_step_train_loss, single_step_val_loss # 能否返回acc？-------------------------


def custompredictX(sess,
                  network,
                  output_provider,
                  x,
                  fragment_size=1000,
                  output_length=1,
                  y_op=None,
                  out_kwag=None):
    """
        Return the predict results of given non time-series network.

        Parameters
        ----------
        sess : TensorFlow session
            sess = tf.InteractiveSession()
        network : a TensorLayer layer
            the network will be trained
        x : placeholder
            the input
        y_op : placeholder
    """

    if y_op is None:
        y_op = network.outputs

    output_container = []
    banum = 0

    for X_out in output_provider.feed(out_kwag['inputs']):
        #log(banum)
        #banum += 1

        feed_dict = {x: X_out,}
        output = sess.run(y_op, feed_dict=feed_dict)
        output_array = np.array(output[0]).reshape(-1, output_length)
        output_array=apply_ema(output_array,1)
        output_container.append(output_array)

    test = np.vstack(output_container)
    return test

def custompredictS2SX(sess,
                  network,
                  output_provider, #传入测试数据集
                  x,
                  fragment_size=1000,
                  output_length=1, #传入windowlength 599
                  y_op=None,
                  out_kwag=None):
    """
        Return the predict results of given non time-series network.

        Parameters
        ----------
        sess : TensorFlow session
            sess = tf.InteractiveSession()
        network : a TensorLayer layer
            the network will be trained
        x : placeholder
            the input
        y_op : placeholder
    """

    if y_op is None:
        y_op = network.outputs  # =network.outputs=y 为预测值

    output_container = None
    for idx, X_out in enumerate(output_provider.feed(out_kwag['inputs'])):  #每次取一定批次的预测数据集
        #log(banum)
        #banum += 1

        feed_dict = {x: X_out,}  # X_out为1000个1*599
        output = sess.run(y_op, feed_dict=feed_dict)
        output_array = np.array(output[0]).reshape(-1, output_length)  #让np.array(output[0])变成只有599列，行自动计算
        if not idx:
            output_container = output_array
        else:
            output_container = np.concatenate((output_container, output_array), axis=0) #axis=0表示列方向上拼接
    l = output_length  # 599
    n = len(output_container) + l - 1
    sum_arr = np.zeros((n))  # 1*n
    counts_arr = np.zeros((n))
    o = len(sum_arr)
    for i in range(len(output_container)):
        sum_arr[i:i + l] += output_container[i].flatten() #默认 按行展平
        counts_arr[i:i + l] += 1
    for i in range(len(sum_arr)):
        sum_arr[i] = sum_arr[i] / counts_arr[i]
    prediction = sum_arr
    return prediction





def custompredict_fcn(sess,
                  network,
                  output_provider,
                  x,
                  fragment_size=1000,
                  output_length=1,
                  y_op=None,
                  out_kwag=None):
    """
        Return the predict results of given non time-series network.

        Parameters
        ----------
        sess : TensorFlow session
            sess = tf.InteractiveSession()
        network : a TensorLayer layer
            the network will be trained
        x : placeholder
            the input
        y_op : placeholder
    """

    if y_op is None:
        y_op = network.outputs

    output_container = []
    banum = 0

    for X_out in output_provider.feed(out_kwag['inputs']):
        #log(banum)
        #banum += 1

        feed_dict = {x: X_out,}
        output = sess.run(y_op, feed_dict=feed_dict)
        output_array = np.array(output[0]).reshape(-1, output_length)
        output_container.append(output_array)

    test = np.vstack(output_container)
    return test


def custompredictS2SXmedian(sess,
                  network,
                  output_provider,
                  x,
                  fragment_size=1000,
                  output_length=1,
                  y_op=None,
                  out_kwag=None):
    """
        Return the predict results of given non time-series network.

        Parameters
        ----------
        sess : TensorFlow session
            sess = tf.InteractiveSession()
        network : a TensorLayer layer
            the network will be trained
        x : placeholder
            the input
        y_op : placeholder
    """

    if y_op is None:
        y_op = network.outputs

    output_container = None
    for idx, X_out in enumerate(output_provider.feed(out_kwag['inputs'])):
        #log(banum)
        #banum += 1

        feed_dict = {x: X_out,}
        output = sess.run(y_op, feed_dict=feed_dict)
        output_array = np.array(output[0]).reshape(-1, output_length)
        if not idx:
            output_container = output_array
        else:
            output_container = np.concatenate((output_container, output_array), axis=0)
    l = output_length
    n = len(output_container) + l - 1
    overlapping = []
    for i in range(n):
        overlapping.append([])

    for i in range(len(output_container)):
        k = 0
        for j in range(i, i+l):
            overlapping[j].append(output_container[i][k])
            k = k+1
    dic_median = pd.DataFrame(overlapping).median(axis=1)
    prediction = np.array(dic_median)
    return prediction

