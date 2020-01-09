from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class L2PGDAttack:
    def __init__(self, model, epsilon, k, a, random_start, loss_func, target_class=None):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.epsilon = epsilon
        self.sqrt_epsilon = np.sqrt(epsilon)
        self.k = k
        self.a = a
        self.rand = random_start

        if loss_func == 'xent':
            loss = model.xent
            self.pre_softmax = model.pre_softmax[:,0]
        elif loss_func == 'cw':
            #model.y_input is the correct label
            #label_mask is a 10*batch matrix
            #value 1 on the correct label, 0 elsewhere
            label_mask = tf.one_hot(model.y_input,
                                    10,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
            #reduce_max returns a batch_size-dim vector,
            #each number being the maximum in the input
            #element-wise multiplication
            #this essentially takes the most correct incorrect class prediction
            wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax
                                        - 1e4*label_mask, axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            loss = model.xent

        self.grad = tf.gradients(loss, model.x_input)[0]
        if(target_class!=None):
            self.grad = tf.gradients(model.pre_softmax[:,target_class], model.x_input)[0]
        self.softmax_grad = tf.gradients(self.pre_softmax, model.x_input)[0]

    def perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
            x = np.clip(x, 0, 1) # ensure valid pixel range
        else:
            x = np.copy(x_nat)
        #grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
        #                                      self.model.y_input: y})
        #softmax_grad = sess.run(self.softmax_grad,
        #                        feed_dict={self.model.x_input: x,
        #                                   self.model.y_input: y})
        #softmax = sess.run(self.pre_softmax,
        #                        feed_dict={self.model.x_input: x,
        #                                   self.model.y_input: y})
        #print('Loss gradients')
        #print(grad)
        #print('Class gradients')
        #print(softmax_grad)
        #print(softmax_grad.shape)
        #print(softmax.shape)

        for i in range(self.k):
            grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                                  self.model.y_input: y})

            #a appears to be similar to a learning-rate
            #it is a constant from the config
            #Perhaps this means that the gradients
            #only determine the direction of change
            #for each pixel, not magnitude.
            x += self.a * np.sign(grad)
            #x += self.a * grad
            #print(np.sum(np.square(x-x_nat)))

            L_2 = np.sum(np.square(x-x_nat),axis=1)
            L2_grad = np.sqrt(np.sum(np.square(x-x_nat),axis=1))*2
            multiplier = 2*np.clip(np.sqrt(L_2)-self.sqrt_epsilon,0,None)/L2_grad
            multiplier = np.array([multiplier]).transpose()
            x -= multiplier*(x-x_nat)
            x = np.clip(x, 0, 1) # ensure valid pixel range

        return x
