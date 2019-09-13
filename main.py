#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow.python.platform
import collections

# -------- File Paths --------

CKPT_PATH = "./model/ai_model.ckpt"
TRAINING_DATA_PATH="./data/train/data.txt"
VALIDATION_DATA_PATH="./data/validate/data.txt"
TRAIN_LOG_PATH="./data/log"

# -------- Hyperparameters --------

# How big should the images be resized into? (px)
# - If you set 16, every image will be resized into 16*16.
IMAGE_SIZE = 28

# How big should the feature point area be? (px)
# - If you set 5, the AI will try to find 5*5 feature point from an image.
PATTERN_SIZE=5

# How many times should the AI learn the images?
MAX_STEPS=100

# How many data should the AI learn per step?
# - BATCH_SIZE must divide the number of training data evenly.
# - If you have 20 training data, BATCH_SIZE must be either 20, 10, 5, 4, 2, or 1.
BATCH_SIZE=50

# How big should the AI change weights of the model every step?
LEARNING_RATE=1e-4

# How many nodes should the AI have for each layer?
NODES={
    0:3,
    1:32,
    2:64
}

# How many nodes should the AI keep?
# If you set 0.9, 10% nodes will be dropped out.
KEEP_PROB=0.4

# Mapping between class number and label.
LABELS = collections.OrderedDict()
LABELS[0]="馬"
LABELS[1]="鹿"


# ----------- Functions ------------

def inference(images_placeholder, keep_prob):

    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def create_conv_layer(image,index):
        with tf.name_scope("conv"+str(index)) as scope:
          W_conv = weight_variable([PATTERN_SIZE, PATTERN_SIZE, NODES[index], NODES[index+1]])
          b_conv = bias_variable([NODES[index+1]])
          h_conv = tf.nn.relu(conv2d(image, W_conv) + b_conv)
          return h_conv

    def create_pool_layer(h_conv,index):
        with tf.name_scope("pool"+str(index)) as scope:
            h_pool = max_pool_2x2(h_conv)
            return h_pool

    h_pool = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, NODES[0]])

    for index in range(len(NODES)-1):
        h_conv=create_conv_layer(h_pool,index)
        h_pool=create_pool_layer(h_conv,index)

    with tf.name_scope("fc1") as scope:
      w = IMAGE_SIZE//4
      last_layer=NODES[len(NODES)-1]
      W_fc1 = weight_variable([w*w*last_layer, 1024])
      b_fc1 = bias_variable([1024])
      h_pool_flat = tf.reshape(h_pool, [-1, w*w*last_layer])
      h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)
      h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope("fc2") as scope:
      W_fc2 = weight_variable([1024, len(LABELS)])
      b_fc2 = bias_variable([len(LABELS)])

    with tf.name_scope("softmax") as scope:
      y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv


def train():

    print("Reading images...")
    def readImagesAndLabels(file_path):
        LINE_INDEX_FILE_NAME=0
        LINE_INDEX_LABEL=1
        images = []
        labels = []
        with open(file_path, "r") as f:
            for line in f:
              l = line.rstrip().split()
              file_name=l[LINE_INDEX_FILE_NAME]
              label=int(l[LINE_INDEX_LABEL])
              img = cv2.imread(file_name)
              img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
              img=img.flatten().astype(np.float32)/255.0
              images.append(img)
              tmp = np.zeros(len(LABELS))
              tmp[label] = 1
              labels.append(tmp)
        return {"images":images,"labels":labels}

    training_data=readImagesAndLabels(TRAINING_DATA_PATH)
    validation_data=readImagesAndLabels(VALIDATION_DATA_PATH)

    with tf.Graph().as_default():

        print("Creating an AI model...")
        def loss(logits, labels):
            cross_entropy = -tf.reduce_sum(labels*tf.log(logits+1e-30))
            tf.summary.scalar("cross_entropy", cross_entropy)
            return cross_entropy

        def training(loss, learning_rate):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            return train_step

        def accuracy(logits, labels):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            tf.summary.scalar("accuracy", accuracy)
            return accuracy

        images_placeholder = tf.placeholder("float", shape=(None, IMAGE_SIZE*IMAGE_SIZE*3))
        labels_placeholder = tf.placeholder("float", shape=(None, len(LABELS)))
        keep_prob = tf.placeholder("float")
        logits = inference(images_placeholder, keep_prob)
        loss_value = loss(logits, labels_placeholder)
        train_op = training(loss_value, LEARNING_RATE)
        acc = accuracy(logits, labels_placeholder)

        train_feed={
          images_placeholder: training_data["images"],
          labels_placeholder: training_data["labels"],
          keep_prob: 1.0}

        test_feed={
          images_placeholder: validation_data["images"],
          labels_placeholder: validation_data["labels"],
          keep_prob: 1.0}

        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(TRAIN_LOG_PATH, sess.graph_def)

        print("Learning started.")
        for step in range(MAX_STEPS):
          for i in range(len(training_data["images"])//BATCH_SIZE):
            batch = BATCH_SIZE*i
            feed={
              images_placeholder: training_data["images"][batch:batch+BATCH_SIZE],
              labels_placeholder: training_data["labels"][batch:batch+BATCH_SIZE],
              keep_prob: KEEP_PROB}
            sess.run(train_op,feed)

          train_accuracy = sess.run(acc, train_feed)
          print ("step %d, training data: accuracy %g"%(step, train_accuracy))

          summary_str = sess.run(summary_op, train_feed)
          summary_writer.add_summary(summary_str, step)

        test_result=sess.run(acc, test_feed)
        print ("validation data: accuracy %g"%test_result)
        saver.save(sess, CKPT_PATH)


def judge(img_path):
    tf.reset_default_graph()
    image = []
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    image.append(img.flatten().astype(np.float32)/255.0)
    image = np.asarray(image)
    logits = inference(image, 1.0)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess, CKPT_PATH)
    softmax = logits.eval()
    softmax_result = softmax[0]
    rates = [round(n * 100.0, 1) for n in softmax_result]
    result = []
    for index, rate in enumerate(rates):
        name = LABELS[index]
        result.append({
          "label": index,
          "name": name,
          "rate": rate
        })
    sorted_result = sorted(result, key=lambda x: x["rate"], reverse=True)
    return sorted_result


if __name__ == "__main__":
    train()
