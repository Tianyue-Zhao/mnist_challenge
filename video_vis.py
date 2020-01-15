from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib
#matplotlib.use('GTKAgg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import re
import tensorflow as tf
from model import Model
import json
from pgd_attack import LinfPGDAttack
from pgd_l2 import L2PGDAttack
from random import random
import cv2
import multiprocessing as mp

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
x=mnist.test.images
y=mnist.test.labels
default_norm = '0' #0 for L2, 1 for L_inf
model_dir = '/home/zhao/Additional_programs/mnist_challenge/models/'
img_dir = '/home/zhao/Additional_programs/mnist_challenge/videos/'
default_steps = 400
default_rate = 0.01
default_epsilon = 784.0
default_examples = 20
default_model_suffix = 'L2_trained'
gpu=tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)
config = json.load(open('config.json'))

#Take inputs
norm = raw_input('Enter 0 for L2 norm, 1 for L_inf norm, default L2\n')
steps = raw_input('Enter the number of steps to perturb, default config\n')
rate = raw_input('Enter the change rate for the perturbation, default config\n')
target_class = raw_input('Enter the target class to perturb to\n')
epsilon = raw_input('Enter the epsilon to use, default '+str(default_epsilon)+'\n')
num_examples = raw_input('Enter the number of examples to generate, default '
                         +str(default_examples)+'\n')
model_suffix = raw_input('Enter the model suffix, default '+default_model_suffix+'\n')

#Process inputs
if(steps==''):
    steps = default_steps
else:
    steps = int(steps)
if(rate==''):
    rate = default_rate
else:
    rate = float(rate)
if(num_examples==''):
    num_examples = default_examples
else:
    num_examples = int(num_examples)
if(model_suffix==''):
    model_suffix = default_model_suffix
if(epsilon==''):
    epsilon = default_epsilon
else:
    epsilon = float(epsilon)
target_class = int(target_class)

used_indices = []
for i in range(num_examples):
    used_indices.append(int(random() * x.shape[0]))
x_orig = x[used_indices, :]
y_orig = y[used_indices]
model = Model()

if((norm=='0')or(norm=='')):
    attack = L2PGDAttack(model,
                         epsilon,
                         steps,
                         rate,
                         False,
                         config['loss_func'],
                         target_class=target_class)
elif(norm=='1'):
    attack = LinfPGDAttack(model,
                           epsilon,
                           steps,
                           rate,
                           False,
                           config['loss_func'])
else:
    print('Unexpected norm selection')
    9/0
checkpoint = tf.train.latest_checkpoint(model_dir+model_suffix)
results = np.zeros((784,num_examples))

#Initialize the matplotlib figure and subplot
fig, (ax) = plt.subplots(1,1)
#Generate the adversarial examples
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,checkpoint)
    x_history, softmax_history = attack.perturb(x_orig,y_orig,sess,return_history=True)

maxlength = len(str(num_examples))
(w,h) = fig.canvas.get_width_height()

#Function for use with parallel-processing
def output_video(cur_index):
    video_out = cv2.VideoWriter(img_dir + ('0' * (maxlength - len(str(cur_index)))) + str(cur_index) + '.avi',
                                cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                                (w, h))
    for k in range(steps + 1):
        ax.imshow(np.stack([np.reshape(x_history[cur_index, :, k], (28, 28))] * 3, -1))
        ax.set_xlabel(str(y_orig[cur_index]) + ': ' + str(softmax_history[cur_index, y_orig[cur_index], k]) + '    '
                      + str(target_class) + ': ' + str(softmax_history[cur_index, target_class, k]))
        fig.canvas.draw()
        frame = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        frame = np.reshape(frame, (h, w, 3))
        video_out.write(frame)

#Utilize multi-processing for the video output
print("Number of processors: ", mp.cpu_count())
pool = mp.Pool(mp.cpu_count())
pool.map(output_video, range(num_examples))
pool.close()
