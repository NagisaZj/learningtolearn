import tensorflow as tf
import tensorlayer as tl
import numpy as np

class grader:
    def __init__(self,hidden_size,type,lr,sess,input):
        self.sess = sess
        self.type = type
        with tf.variable_scope(type,reuse=tf.AUTO_REUSE) as scope:
            self.scope = type
            self.cell_list = []
            self.rnn = tl.layers.InputLayer(input,name="in")
            self.rnn = tl.layers.RNNLayer(self.rnn,cell_fn = tf.nn.rnn_cell.BasicLSTMCell,n_hidden = hidden_size,n_steps = 1,return_last = False,return_seq_2d = False,name="rnn1")
            self.rnn = tl.layers.RNNLayer(self.rnn,cell_fn = tf.nn.rnn_cell.BasicLSTMCell,n_hidden = hidden_size,n_steps = 1,return_last = False,return_seq_2d = True,name = "rnn2")
            self.rnn = tl.layers.DenseLayer(self.rnn,n_units=1,name = "dense_opti")
            self.output = self.rnn.outputs
            #self.W = tf.get_variable("W",[hidden_size,1],dtype=tf.float32)
            #self.b = tf.get_variable("b",[1],dtype = tf.float32)
            #self.tvars =  tf.trainable_variables()
            self.tvars = self.rnn.all_params
            print(self.tvars)
            self.optimizer = tf.train.AdamOptimizer(lr)

    def feed(self,input,state):
        with tf.variable_scope(self.type,reuse=tf.AUTO_REUSE) as scope:
            self.output,state_after = tf.nn.dynamic_rnn(self.cell, input,initial_state=state, time_major=False)
            self.out = tf.matmul(self.output[:,-1,:],self.W) +self.b
            return self.out,state_after

    def train(self,loss):
        with tf.variable_scope(self.type,reuse=tf.AUTO_REUSE) as scope:
            grads = tf.gradients(loss, self.tvars)
            self.train_op = self.optimizer.apply_gradients(zip(grads, self.tvars))

    def save(self):
        tl.files.exists_or_mkdir(self.scope)
        tl.files.save_ckpt(sess=self.sess, mode_name='model.ckpt', var_list=self.tvars,
                           save_dir=self.scope, printable=False)

    def load(self):
        tl.files.load_ckpt(sess=self.sess, var_list=self.tvars, save_dir=self.scope, printable=False)



