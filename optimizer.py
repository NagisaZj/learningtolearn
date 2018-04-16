import tensorflow as tf
import tensorlayer as tl
import numpy as np

class grader:
    def __init__(self,hidden_size,layers,batch_size,num,lr):
        with tf.variable_scope("grader%d"%num) as scope:
            self.cell_list = []
            for i in range(layers):
                self.cell_list.append(tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True))
            self.cell = tf.nn.rnn_cell.MultiRNNCell(self.cell_list, state_is_tuple=True)
            self.state = cell.zero_state(batch_size, tf.float32)
            self.W = tf.get_variable("W",[hidden_size,1],dtype=tf.float32)
            self.b = tf.get_variable("b",[1],dtype = tf.float32)
            self.tvars =  tf.trainable_variables()
            self.optimizer = tf.train.AdamOptimizer(lr)

    def feed(self,input):
        self.output,self.state = tf.nn.dynamic_rnn(cell, input,initial_state=state, time_major=False)
        self.out = tf.matmul(self.output[:,-1,:],self.W) +self.b
        return self.out

    def train(self,loss):
        grads = tf.gradients(loss, tvars)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))


