import csv
import os
import tensorflow as tf
from PIL import Image

DTRAIN = "train/image-data/train"
DTEST = "test/image-data/test"
CSVTRAIN = "train/image-data/train.csv"
CSVTEST = "test/prediction.csv"

CLASS_SIZE = 16
BATCH_SIZE = 10
TEST_SIZE = 4138

WIDTH = 84
HEIGHT = 96
CHANNELS = 3

ITERATION = 101
KEEP_NUM = 0.3 # the probability of each neuron being kept

def read_data_queue(directory, name):
    filelist = []

    for index, filename in enumerate(os.listdir(directory)):
        filelist.append(os.path.join(directory, filename))

    filelist.sort(key=lambda f: int(filter(str.isdigit, f)))

    labellist = []

    for row in csv.reader(open(name)):
        if len(row) > 1:
            labellist.append(int(row[1]))
        else:
            labellist.append(0)

    labels = tf.one_hot(labellist, CLASS_SIZE)
    images = tf.convert_to_tensor(filelist, dtype=tf.string)
    
    return labels, images

def read_data_batch(labellist, filelist):
    input_queue = tf.train.slice_input_producer([labellist, filelist])
    labels = input_queue[0]
    imagelist = tf.read_file(input_queue[1])
    images = tf.image.resize_images(tf.image.decode_jpeg(imagelist), [WIDTH, HEIGHT])
    images.set_shape([WIDTH, HEIGHT, CHANNELS])
    return labels, images

def next_batch(labels, images, batch_size = BATCH_SIZE):
    label_batch, image_batch = tf.train.batch([labels, images], batch_size = batch_size)
    return label_batch, image_batch

def weight_variable(shape): # random weight
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1))

def bias_variable(shape): # random bias
    return tf.Variable(tf.constant(0.1, shape = shape))

def conv2d(x, w, strides_size, padding_type = "VALID"): # convolution
    return tf.nn.conv2d(x, w, strides = [1, strides_size, strides_size, 1], padding = padding_type)

def max_pooling(x, k_size, strides_size, padding_type = "VALID"): # pooling
    return tf.nn.max_pool(x, ksize = [1, k_size, k_size, 1], strides = [1, strides_size, strides_size, 1], padding = padding_type)

def CNNet(x, keep_prob):
    # input layer
    x_input = tf.reshape(x, [-1, WIDTH, HEIGHT, CHANNELS]) # 84 * 96 * 1

    # hidden layer 1
    w_conv1 = weight_variable([3, 3, CHANNELS, 16])
    conv1 = tf.nn.conv2d(x_input, w_conv1, strides = [1, 1, 1, 1], padding = "VALID") # 82 * 94 * 16
    b_conv1 = bias_variable([16])
    relu1 = tf.nn.relu(conv1 + b_conv1)

    # hidden layer 2
    w_conv2 = weight_variable([3, 3, 16, 4]) 
    conv2 = tf.nn.conv2d(relu1, w_conv2, strides = [1, 1, 1, 1], padding = "VALID") # 80 * 92 * 4
    b_conv2 = bias_variable([4])
    relu2 = tf.nn.relu(conv2 + b_conv2)

    # hidden layer 3
    w_conv3 = weight_variable([3, 3, 4, 8])
    conv3 = tf.nn.conv2d(relu2, w_conv3, strides = [1, 1, 1, 1], padding = "VALID") # 78 * 90 * 8
    pool1 = max_pooling(conv3, 3, 3, "SAME") # 26 * 30 * 8

    # hidden layer 4
    w_conv4 = weight_variable([3, 3, 8, 16])
    conv4 = tf.nn.conv2d(pool1, w_conv4, strides = [1, 1, 1, 1], padding = "VALID") # 24 * 28 * 16
    pool2 = max_pooling(conv4, 3, 3, "SAME") # 8 * 10 * 16

    # fully connected layer 1
    pool2_flat = tf.reshape(pool2, [-1, 8 * 10 * 16])
    w_fc1 = weight_variable([8 * 10 * 16, 512])
    b_fc1 = bias_variable([512])
    fc1 = tf.matmul(pool2_flat, w_fc1) + b_fc1
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

    # fully connected layer 2
    w_fc2 = weight_variable([512, CLASS_SIZE])
    b_fc2 = bias_variable([CLASS_SIZE])
    fc2 = tf.matmul(fc1_drop, w_fc2) + b_fc2

    # output layer
    # prediction = tf.nn.softmax(fc2)

    # return prediction
    return fc2

def main():
    train_label_list, train_image_list = read_data_queue(DTRAIN, CSVTRAIN)
    test_label_list, test_image_list = read_data_queue(DTEST, CSVTEST)

    train_label, train_image = read_data_batch(train_label_list, train_image_list)
    test_label, test_image = read_data_batch(test_label_list, test_image_list)

    train_label_batch, train_image_batch = next_batch(train_label, train_image, BATCH_SIZE)
    test_label_batch, test_image_batch = next_batch(test_label, test_image, TEST_SIZE)
    
    keep_prob = tf.placeholder(tf.float32)
    x = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, CHANNELS])
    y = tf.placeholder(tf.float32, [None, CLASS_SIZE])
    
    y_prediction = CNNet(x, keep_prob)

    with tf.name_scope('loss'):
        cost = tf.reduce_sum(tf.losses.sigmoid_cross_entropy(y, y_prediction)) # loss
        train_step = tf.train.AdamOptimizer(1e-3).minimize(cost) # optimizer
        loss_summary = tf.summary.scalar('loss', cost)
        
    with tf.name_scope('accuracy'):
        result = tf.argmax(y_prediction, 1)
        correct = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    
    
    init_g = tf.global_variables_initializer()
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))) as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs/", sess.graph)

        sess.run(init_g)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # iteration
        for i in range(ITERATION):
            if coord.should_stop():
                print('corrd break!!!!!!')
                break
            train_label_next, train_image_next = sess.run([train_label_batch, train_image_batch])
            _, loss = sess.run([train_step, cost], feed_dict={x: train_image_next, y: train_label_next, keep_prob: KEEP_NUM})

            if (i % 100 == 0) & (i != 0):
                test_label_next, test_image_next = sess.run([test_label_batch, test_image_batch])
                test_acc = sess.run(accuracy, feed_dict={
                                    x: test_image_next, y: test_label_next, keep_prob: 1})
                test_res = sess.run(result, feed_dict={
                                    x: test_image_next, y: test_label_next, keep_prob: 1})



                print('iter: ', i)
                print('loss: ', loss)
                print('test_res: ', test_res)

                saver = tf.train.Saver()
                saver.save(sess, 'final_project', global_step=i)

                with open("prediction.csv", "wb") as f:
                    writer = csv.writer(f, dialect='excel')
                    writer.writerows([test_res])

        coord.request_stop()
        coord.join(threads)

main()
