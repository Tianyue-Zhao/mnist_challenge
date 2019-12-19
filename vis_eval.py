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
model_dir = '/Users/zao/Additional_applications/mnist_challenge/models/'
img_dir = '/Users/zao/Additional_applications/mnist_challenge/comparison_imgs/'

suffix1 = raw_input('Enter the first model directory, default config')
suffix2 = raw_input('Enter the second model directory, default config')
img_suffix = raw_input('Enter the output img folder suffix, default none')
num_examples = raw_input('Enter the number of examples to generate, default '
                         +str(default_num))
epsilon = raw_input('Enter the epsilon to use, default config value')

config = json.load(open('config.json'))

#Handle the input values
if(img_suffix!=''):
    img_dir += img_suffix+'/'

if(suffix1==''):
    suffix1=config['suffix1']
if(suffix2==''):
    suffix2=config['suffix1']
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
attack = LinfPGDAttack(model,
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
    saver.restore(sess,checkpoint2)
    x_adv_2 = attack.perturb(x_orig,y_orig,sess)

#Declare the matplotib plots to use
fig,(orig_ax,adv_1_ax,adv_2_ax) = plt.subplots(1,3)
maxlength=len(str(num_examples))

for i in range(num_examples):
    orig_ax.imshow(np.stack([np.reshape(x_orig[i,:],(28,28))]*3,-1))
    adv_1_ax.imshow(np.stack([np.reshape(x_adv_1[i,:],(28,28))]*3,-1))
    adv_2_ax.imshow(np.stack([np.reshape(x_adv_2[i,:],(28,28))]*3,-1))
    plt.savefig(img_dir+('0'*(maxlength-len(str(i))))+str(i)+'.png')
