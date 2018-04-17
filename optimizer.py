import tensorflow as tf
import tensorlayer as tl
import numpy as np

class grader:
    def __init__(self,hidden_size,layers,batch_size,type,lr,sess):
        self.sess = sess
        self.num = num
        with tf.variable_scope(type,reuse=tf.AUTO_REUSE) as scope:
            self.scope = type
            self.cell_list = []
            for i in range(layers):
                self.cell_list.append(tf.nn.rnn_cell.BasicLSTMCell(hidden_size, state_is_tuple=True))
            self.cell = tf.nn.rnn_cell.MultiRNNCell(self.cell_list, state_is_tuple=True)
            self.state = self.cell.zero_state(batch_size, tf.float32)
            self.W = tf.get_variable("W",[hidden_size,1],dtype=tf.float32)
            self.b = tf.get_variable("b",[1],dtype = tf.float32)
            self.tvars =  tf.trainable_variables()
            self.optimizer = tf.train.AdamOptimizer(lr)

    def feed(self,input):
        with tf.variable_scope("grader%d" % self.num) as scope:
            self.output,self.state = tf.nn.dynamic_rnn(self.cell, input,initial_state=self.state, time_major=False)
            self.out = tf.matmul(self.output[:,-1,:],self.W) +self.b
            return self.out

    def train(self,loss):
        with tf.variable_scope("grader%d" % self.num) as scope:
            grads = tf.gradients(loss, self.tvars)
            self.train_op = self.optimizer.apply_gradients(zip(grads, self.tvars))

    def save(self):
        tl.files.exists_or_mkdir(self.scope)
        tl.files.save_ckpt(sess=self.sess, mode_name='model.ckpt', var_list=self.tvars,
                           save_dir=self.scope, printable=False)

    def load(self):
        tl.files.load_ckpt(sess=self.sess, var_list=self.tvars, save_dir=self.scope, printable=False)



