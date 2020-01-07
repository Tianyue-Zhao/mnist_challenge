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

#Take input: checkpoint 1, default 0
#Take input: checkpoint 2, default last
#Take input: alternate folder for model?
#Take input: alternate folder for pictures?
#Take input: how many examples? Default?
#Also save npy files
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
print('The shape of the images array is')
print(mnist.test.images.shape)
x=mnist.test.images
y=mnist.test.labels
original_img = np.reshape(x[0,:],(28,28))
original_img = np.stack([original_img]*3,-1)
img_show = plt.imshow(original_img)
plt.savefig('original_img.png')
print('The corresponding label is')
print(mnist.test.labels[0])

def getkey(ckptname):
    ckptname = ckptname.strip('checkpoint-')
    ckptname = re.sub(r'\.data*','',ckptname)
    return int(ckptname)

#Take input options
default_model = 'a_very_robust_model'
default_num = 20
model_dir = '/home/zhao/Additional_programs/mnist_challenge/models/'
img_dir = '/home/zhao/Additional_programs/mnist_challenge/comparison_imgs/'

suffix1 = raw_input('Enter the first model directory, default config')
suffix2 = raw_input('Enter the second model directory, default config')
img_suffix = raw_input('Enter the output img folder suffix, default none')
num_examples = raw_input('Enter the number of examples to generate, default '
                         +str(default_num))
epsilon = raw_input('Enter the epsilon to use, default config value')

config = json.load(open('config.json'))
gpu=tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

#Handle the input values
if(img_suffix!=''):
    img_dir += img_suffix+'/'

if(suffix1==''):
    suffix1=config['suffix1']
if(suffix2==''):
    suffix2=config['suffix2']
checkpoint1 = tf.train.latest_checkpoint(model_dir+suffix1)
checkpoint2 = tf.train.latest_checkpoint(model_dir+suffix2)
if((checkpoint1==None) or (checkpoint2==None)):
    print("At least one model suffix is not valid")
    print(9/0)

if(num_examples==''):
    num_examples = default_num
else:
    num_examples = int(num_examples)

if(epsilon==''):
    epsilon = config['epsilon']
else:
    epsilon = float(epsilon)
    #Get the max checkpoint
    #A=glob.glob(model_dir+'/checkpoint-*\.data*')
    #maxckpt = 0
    #for i in range(0,len(A)):
    #    curkey = getkey(A[i])
    #    if(curkey>maxckpt):
    #        maxckpt = curkey
    #checkpoint2 = str(maxckpt)

#Extract random examples to use
used_indexes=[]
for i in range(num_examples):
    used_indexes.append(int(random()*x.shape[0]))
x_orig = x[used_indexes,:]
y_orig = y[used_indexes]

#Declare the model and attacks to use
model = Model()
attack = L2PGDAttack(model,
                       epsilon,
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

#Generate the adversarial examples
#Should add functionality to display the new y class
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,checkpoint1)
    x_adv_1 = attack.perturb(x_orig,y_orig,sess)
    res_pre_1 = sess.run(model.pre_softmax,
                         feed_dict={model.x_input: x_orig})
    arg_pre_1 = np.argmax(res_pre_1, axis=1)
    res_post_1 = sess.run(model.pre_softmax,
                          feed_dict={model.x_input: x_adv_1})
    arg_post_1 = np.argmax(res_post_1, axis=1)
    saver.restore(sess,checkpoint2)
    x_adv_2 = attack.perturb(x_orig,y_orig,sess)
    res_pre_2 = sess.run(model.pre_softmax,
                         feed_dict={model.x_input: x_orig})
    arg_pre_2 = np.argmax(res_pre_2, axis=1)
    res_post_2 = sess.run(model.pre_softmax,
                          feed_dict={model.x_input: x_adv_2})
    arg_post_2 = np.argmax(res_post_2, axis=1)

#Declare the matplotib plots to use
fig,(orig_ax,adv_1_ax,adv_2_ax) = plt.subplots(1,3)
maxlength=len(str(num_examples))
fig.suptitle("Epsilon: "+str(epsilon))
orig_ax.set_title("Original example")
adv_1_ax.set_title(suffix1)
adv_2_ax.set_title(suffix2)
print(orig_ax.title)

for i in range(num_examples):
    adv_1_ax.set_xlabel(str(arg_post_1[i]))
    adv_2_ax.set_xlabel(str(arg_post_2[i]))
    orig_ax.imshow(np.stack([np.reshape(x_orig[i,:],(28,28))]*3,-1))
    adv_1_ax.imshow(np.stack([np.reshape(x_adv_1[i,:],(28,28))]*3,-1))
    adv_2_ax.imshow(np.stack([np.reshape(x_adv_2[i,:],(28,28))]*3,-1))
    plt.savefig(img_dir+('0'*(maxlength-len(str(i))))+str(i)+'.png')
