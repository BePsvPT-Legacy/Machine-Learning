import os
import glob
import tensorflow as tf
from PIL import Image

DATA_DIR = "source" # original dataset folder name
PNG_DATA_DIR = "data" # new dataset folde name
NUM = 13 # each class train data number
WIDTH = 168
HEIGHT = 192
CHANNELS = 1
NUM_CLASSES = 38 # output classes number
BATCH_SIZE = 10 # batch size
ITERATION = 10000 # iteration number
TEST_NUM = 650 # test data number

def Image2PNG():
    for index, classes in enumerate(os.listdir(DATA_DIR)): # read all object name in DATA_DIR to classes
        fullPath = os.path.join(DATA_DIR, classes, "") # combine PATH

        if os.path.isdir(fullPath): # if PATH is not a folder then continue
            pass
        else:
            continue

        globList = glob.glob(fullPath+'*.pgm') # find all filename extension is .bmp to globList
        length_dir = len(fullPath) # cal length of fullPath

        i = 0
        for glob_path in globList:
            i = i + 1

            target = '/test/'
            if i >= NUM: # save to another folder become a new dataset
                target = '/train/'

            Image.open(glob_path).save(PNG_DATA_DIR+target+str(int(glob_path[length_dir-3:length_dir-1])-1).zfill(3)+'-label_'+str(i).zfill(2)+'-num'+'.png')

def read_DatasList(data_type):
    fullLabel = []
    fullPath = os.path.join(PNG_DATA_DIR, data_type, "")

    globList = glob.glob(fullPath+'*.png')
    length_dir = len(fullPath)

    for glob_path in globList:
        fullLabel.append(int(glob_path[length_dir:length_dir+3])) # append all label to fullList

    images = tf.convert_to_tensor(globList, dtype=tf.string)
    labels = tf.one_hot(fullLabel, NUM_CLASSES) # change label to one hot type

    return images, labels

def read_Data_to_batch(fullList, fullLabel):
    input_queue = tf.train.slice_input_producer([fullList, fullLabel])

    file_content = tf.read_file(input_queue[0]) # read PATH

    imgs = tf.image.resize_images(tf.image.decode_png(file_content, CHANNELS), [WIDTH, HEIGHT])

    tf.image.per_image_standardization(imgs)

    labels = input_queue[1]

    return imgs, labels

def next_batch(img, label, batch_size=BATCH_SIZE):
    image_batch, label_batch = tf.train.batch([img, label], batch_size=batch_size)

    return image_batch, label_batch

def CNNet(data, mode):
    # input layer
    input_layer = tf.reshape(data, [-1, WIDTH, HEIGHT, CHANNELS])

    # convolutional layer #1
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)

    # pooling layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=2, strides=2)

    # convolutional layer #2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)

    # pooling layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=2, strides=2)

    # dense Layer
    pool2_flat = tf.reshape(pool2, [-1, (WIDTH / 4) * (HEIGHT / 4) * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.3, training=mode == 'train')

    # logits Layer
    logits = tf.layers.dense(inputs=dropout, units=NUM_CLASSES)

    return tf.nn.softmax(logits)

def main():
    # read datalist or CSV
    fullList_train, fullLabel_train = read_DatasList('train')
    fullList_test, fullLabel_test = read_DatasList('test')

    # read data to a slice/batch
    train_img, train_label = read_Data_to_batch(fullList_train, fullLabel_train)
    test_img, test_label = read_Data_to_batch(fullList_test, fullLabel_test)

    # read the next batch
    train_image_batch, train_label_batch = next_batch(train_img, train_label, BATCH_SIZE)
    test_image_batch, test_label_batch = next_batch(test_img, test_label, TEST_NUM)

    # placeholder
    data = tf.placeholder(tf.float32, [None, WIDTH, HEIGHT, CHANNELS])
    mode = tf.placeholder(tf.string)
    y = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    # call to Net
    y_prediction = CNNet(data, mode)

    # loss
    cost = tf.reduce_sum(tf.losses.softmax_cross_entropy(y, y_prediction)) # loss
    train_step = tf.train.AdamOptimizer(0.0005).minimize(cost) # optimizer

    # accuracy
    correct = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # initializer
    init_g = tf.global_variables_initializer()

    # session
    with tf.Session() as sess:
        # initialize variables
        sess.run(init_g)

        # create a corrdinator
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # iteration
        for i in range(ITERATION):
            if coord.should_stop():
                print('corrd break!!!!!!')
                break

            # load train data batch
            example_train, l_train = sess.run([train_image_batch, train_label_batch])
            # train
            _, loss = sess.run([train_step, cost], feed_dict={data: example_train, y: l_train, mode: 'train'})

            if (i % 100 == 0) & (i != 0):
                # load test data batch
                example_test, l_test = sess.run([test_image_batch, test_label_batch])
                # test accuracy
                test_acc = sess.run(accuracy, feed_dict={data: example_test, y: l_test, mode: 'false'})

                print('iter: ', i)
                print('loss: ', loss)
                print('test_acc: ', test_acc)
        coord.request_stop()
        coord.join(threads)

if tf.gfile.Exists(PNG_DATA_DIR): # if the path already exists do nothing
    pass
else:
    tf.gfile.MakeDirs(PNG_DATA_DIR)
    tf.gfile.MakeDirs(PNG_DATA_DIR+'/train')
    tf.gfile.MakeDirs(PNG_DATA_DIR+'/test')
    Image2PNG()

main()
