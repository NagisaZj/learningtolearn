import numpy as np
import tensorflow as tf
import tensorlayer as tl
from optimizer import grader
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

n_dimension = 784
net_size = 200
hidden_size = 10
layers = 2
#batch_size = 1
lr = 1e-3
full_batch = 128
train_steps = 10000
sess = tf.Session()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print (mnist.train.images.shape)

class train:
    def __init__(self,sess):
        self.sess = sess
        self.state_sigmoid = []
        self.state_softmax = []
        self.update_sigmoid = []
        self.update_softmax = []
        self.build_whole()
        return

    def build_whole(self):
        self.build_target_net(0)
        self.out_grads()
        self.build_opti()
        self.apply_grads()
        #self.update()
        tl.layers.initialize_global_variables(sess)

    def build_target_net(self,times):
        with tf.variable_scope("optimizee%d"%times) as scope:
            w_init = tf.contrib.layers.xavier_initializer()
            self.input = tf.placeholder(tf.float32,[None,n_dimension])
            net = tl.layers.InputLayer(self.input,"In")
            net = tl.layers.DenseLayer(net,n_units=net_size,act = tf.nn.sigmoid,W_init = w_init, name="sigmoid1")
            net = tl.layers.DenseLayer(net,n_units=10,act =tf.nn.softmax,W_init = w_init, name="softmax1")
            self.output = net.outputs
            self.label = tf.placeholder(tf.float32,[None,10])
            loss = tf.reduce_mean(tf.square(self.output-self.label))
            self.loss = loss
            self.params = net.all_params
            self.sigmoid_params = tl.layers.get_variables_with_name("optimizee%d/sigmoid1"%times)
            self.softmax_params = tl.layers.get_variables_with_name("optimizee%d/softmax1"%times)
            self.gradients = tf.gradients(loss,self.params)
            self.sigmoid_gradients = tf.gradients(loss,self.sigmoid_params)
            self.softmax_gradients = tf.gradients(loss, self.softmax_params)
            #self.optimizer = tf.train.AdamOptimizer(1e-3)
            #self.loss_op = self.optimizer.minimize(self.loss)


    def save_ckpt(self):
        tl.files.exists_or_mkdir("model")
        tl.files.save_ckpt( sess=self.sess, mode_name='model.ckpt', var_list=self.params,
                           save_dir="model", printable=False)

    def load_ckpt(self):
        tl.files.load_ckpt(sess=self.sess, var_list=self.params, save_dir="model", printable=False)


    def build_opti(self):
        self.sigmoid_num = 0
        self.sigmoid_optimizer = grader(hidden_size, "sd", lr, self.sess,self.grad_sigmoid)
        self.sigmoid_optimizer.load()
        for param in self.sigmoid_params:
            if len(param.shape) == 1:
                self.sigmoid_num = self.sigmoid_num + int(param.shape[0])
            elif len(param.shape) == 2:
                self.sigmoid_num = self.sigmoid_num + int(param.shape[0]) * int(param.shape[1])
        #self.state_sigmoid = self.sigmoid_optimizer.cell.zero_state(self.sigmoid_num, tf.float32)
        #print(self.state_sigmoid)


        self.softmax_num = 0
        self.softmax_optimizer = grader(hidden_size, "sx", lr, self.sess,self.grad_softmax)
        self.softmax_optimizer.load()
        for param in self.softmax_params:
            if len(param.shape) == 1:
                self.softmax_num = self.softmax_num + int(param.shape[0])
            elif len(param.shape) == 2:
                self.softmax_num = self.softmax_num + int(param.shape[0]) * int(param.shape[1])
        #self.state_softmax = self.softmax_optimizer.cell.zero_state(self.softmax_num, tf.float32)





    def out_grads(self):
        num = 0
        grad_sigmoid = None
        grad_softmax = None
        for grad in self.sigmoid_gradients:
            if len(grad.shape) == 1:
                grad_sigmoid = grad if grad_sigmoid is None else tf.concat([grad_sigmoid,grad],0)

            elif len(grad.shape) == 2:
                for j in range(grad.shape[1]):
                    grad_sigmoid = grad[:,j] if grad_sigmoid is None else tf.concat([grad_sigmoid, grad[:,j]],0)

        self.grad_sigmoid = tf.reshape(grad_sigmoid,[-1,1,1])
        #self.update_sigmoid = self.sigmoid_optimizer.output

        for grad in self.softmax_gradients:
            if len(grad.shape) == 1:
                grad_softmax = grad if grad_softmax is None else tf.concat([grad_softmax,grad],0)
            elif len(grad.shape) == 2:
                for j in range(grad.shape[1]):
                    grad_softmax = grad[:,j] if grad_softmax is None else tf.concat([grad_softmax, grad[:,j]],0)

        self.grad_softmax = tf.reshape(grad_softmax,[-1,1,1])
        #self.update_softmax = self.softmax_optimizer.output
        
        #self.update_sigmoid = tf.reshape(self.update_sigmoid,[self.update_sigmoid.shape[0]])
        #self.update_softmax = tf.reshape(self.update_softmax, [self.update_softmax.shape[0]])

    def apply_grads(self):
        self.update_sigmoid = self.sigmoid_optimizer.output
        self.update_softmax = self.softmax_optimizer.output
        self.update_sigmoid = tf.reshape(self.update_sigmoid,[self.update_sigmoid.shape[0]])
        self.update_softmax = tf.reshape(self.update_softmax,[self.update_softmax.shape[0]])
        self.apply_grad_op = []
        num = 0
        for param in self.sigmoid_params:
            if len(param.shape) == 1:
                self.apply_grad_op.append(tf.assign(param, param + self.update_sigmoid[num:num + int(param.shape[0])]))
                num = num + int(param.shape[0])

            elif len(param.shape) == 2:
                param= param + tf.reshape(tf.stack(self.update_sigmoid[num: num + int(param.shape[0]*int(param.shape[1]))]),[param.shape[0],param.shape[1]])
                num = num + int(param.shape[0])*int(param.shape[1])
        num = 0
        for param in self.softmax_params:
            if len(param.shape) == 1:
                self.apply_grad_op.append(tf.assign(param, param + self.update_softmax[num:num + int(param.shape[0])]))
                num = num + int(param.shape[0])

            elif len(param.shape) == 2:
                for i in range(param.shape[1]):
                    self.apply_grad_op.append(tf.assign(param[:,i], param[:,i] + tf.stack(self.update_softmax[num: num + int(param.shape[0])])))
                    num = num + int(param.shape[0])
        self.loss = tf.reduce_mean(tf.square(self.output-self.label))

        #self.update_sigmoid = None
        #self.update_softmax = None

    def update(self):

        self.sigmoid_optimizer.train(self.loss)
        self.softmax_optimizer.train(self.loss)


    def save_opti(self):
        self.sigmoid_optimizer.save()
        self.softmax_optimizer.save()

    def train_one_fun(self):
        #self.build_whole()
        losses = []
        for i in range(train_steps):
            mini_batch = mnist.train.next_batch(full_batch)
            W = mini_batch[0]
            y = mini_batch[1]
            feed_dict = {self.input: W, self.label: y}
            loss,sigmoid_gradients,softmax_gradients= self.sess.run([self.loss,self.sigmoid_gradients,self.softmax_gradients],feed_dict = feed_dict)
            #self.out_grads(sigmoid_gradients,softmax_gradients)
            self.sess.run([self.apply_grad_op],feed_dict = feed_dict)
            #self.apply_grads()
            self.update()
            self.sess.run([self.sigmoid_optimizer.train_op],feed_dict = feed_dict)
            self.sess.run([self.softmax_optimizer.train_op], feed_dict=feed_dict)
            if i %10 ==0:
                print(loss)
                losses.append(loss)
        l = np.array(losses, dtype=np.float32)
        l.tofile("rnn.bin")
        #plt.plot(losses)
        #plt.savefig("rnn2.jpg")

    def train_contrast(self):
        #self.build_target_net(0)
        losses = []
        for i in range(train_steps):
            mini_batch = mnist.train.next_batch(full_batch)
            W = mini_batch[0]
            y = mini_batch[1]
            feed_dict = {self.input: W, self.label: y}
            loss,_ = self.sess.run([self.loss,self.loss_op], feed_dict=feed_dict)
            if i % 10 == 0:
                print(loss)
                losses.append(loss)

        l = np.array(losses, dtype=np.float32)
        l.tofile("adam.bin")
        #plt.plot(losses)
        #plt.savefig("adam2.jpg")
        #print(W)


trainer = train(sess)
#trainer.load_ckpt()
#trainer.train_contrast()
trainer.train_one_fun()
#trainer.save_ckpt()
trainer.save_opti()
#optimizer_0 = grader(hidden_size,layers,batch_size,0,lr)
#optimizer_0.feed(tf.reshape(params[0][0][0],[1,1,1]))



