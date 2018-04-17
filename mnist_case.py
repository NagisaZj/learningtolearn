import numpy as np
import tensorflow as tf
import tensorlayer as tl
from optimizer import grader
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

n_dimension = 784
net_size = 300
hidden_size=10
layers = 2
batch_size = 1
lr = 1e-3
full_batch = 128
train_steps = 1000
sess = tf.Session()

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print (mnist.train.images.shape)

class train:
    def __init__(self,sess):
        self.optimizers = []
        self.sess = sess
        self.grad_list = []
        self.params = []
        self.build_whole()
        return

    def build_whole(self):
        self.build_target_net(0)
        self.build_opti()
        self.out_grads()
        self.apply_grads()
        self.update()
        tl.layers.initialize_global_variables(sess)

    def build_target_net(self,times):
        with tf.variable_scope("optimizee%d"%times) as scope:
            w_init = tf.contrib.layers.xavier_initializer()
            self.input = tf.placeholder(tf.float32,[None,n_dimension])
            net = tl.layers.InputLayer(self.input,"In")
            net = tl.layers.DenseLayer(net,n_units=net_size,act = tf.nn.sigmoid,W_init = w_init, name="sigmoid1")
            net = tl.layers.DenseLayer(net,n_units=10,act =tf.nn.softmax,W_init = w_init, name="softmax1")
            output = net.outputs
            self.label = tf.placeholder(tf.float32,[None,10])
            loss = tf.reduce_mean(tf.square(output-self.label))
            self.loss = loss
            self.params = net.all_params
            self.sigmoid_params = tl.layers.get_variables_with_name("optimizee%d/sigmoid1"%times)
            self.softmax_params = tl.layers.get_variables_with_name("optimizee%d/softmax1"%times)
            self.gradients = tf.gradients(loss,self.params)
            self.sigmoid_gradients = tf.gradients(loss,self.sigmoid_params)
            self.softmax_gradients = tf.gradients(loss, self.softmax_params)
            self.optimizer = tf.train.AdamOptimizer(1e-3)
            self.loss_op = self.optimizer.minimize(self.loss)


    def opti_util(self,param,num,type):
        if len(param.shape) == 1:
            for i in range(param.shape[0]):
                gra = grader(hidden_size, layers, batch_size, type, lr, self.sess)
                gra.load()
                self.optimizers.append(gra)
                num = num + 1
        elif len(param.shape) == 2:
            for i in range(param.shape[0]):
                for j in range(param.shape[1]):
                    gra = grader(hidden_size, layers, batch_size, type, lr, self.sess)
                    gra.load()
                    self.optimizers.append(gra)
                    num = num + 1
        return num


    def build_opti(self):
        num = 0
        for param in self.sigmoid_params:
            num = self.opti_util(param,num,"sigmoid")
            self.sigmoid_opti = self.optimizers[num-1]

        for param in self.softmax_params:
            num = self.opti_util(param,num,"softmax")
            self.softmax_opti = self.optimizers[num - 1]

    def grad_util(self,grad,num):
        if len(grad.shape) == 1:
            for i in range(grad.shape[0]):
                gra = self.optimizers[num].feed(tf.reshape(grad[i], [1, 1, 1]))
                self.grad_list.append(gra)
                num = num + 1

        elif len(grad.shape) == 2:
            for i in range(grad.shape[0]):
                for j in range(grad.shape[1]):
                    gra = self.optimizers[num].feed(tf.reshape(grad[i, j], [1, 1, 1]))
                    self.grad_list.append(gra)
                    num = num + 1
        return num

    def out_grads(self):
        num = 0
        for grad in self.sigmoid_gradients:
            num = self.grad_util(grad,num)

        for grad in self.softmax_gradients:
            num = self.grad_util(grad,num)


    def apply_util(self,param,num):
        if len(param.shape) == 1:
            for i in range(param.shape[0]):
                tf.assign(param[i], param[i] + self.grad_list[num])
                num = num + 1

        elif len(param.shape) == 2:
            for i in range(param.shape[0]):
                tf.assign(param[i], param[i] + tf.stack(self.grad_list[num: num + int(param.shape[1])]))
                num = num + int(param.shape[1])
        return num


    def apply_grads(self):
        num = 0
        for param in self.sigmoid_params:
            num = self.apply_util(param,num)
        for param in self.softmax_params:
            num = self.apply_util(param,num)

        self.grad_list = []

    def update(self):
        for opt in self.optimizers:
            opt.train(self.loss)


    def save_opti(self):
        self.sigmoid_opti.save()
        self.softmax_opti.save()


    def train_one_fun(self):
        losses = []
        for i in range(train_steps):
            mini_batch = mnist.train.next_batch(full_batch)
            W = mini_batch[0]
            y = mini_batch[1]
            feed_dict = {self.input: W, self.label: y}
            loss= self.sess.run([self.loss],feed_dict = feed_dict)
            for opt in self.optimizers:
                self.sess.run([opt.train_op],feed_dict = feed_dict)
            if i %10 ==0:
                print(loss)
                losses.append(loss)
        l = np.array(losses, dtype=np.float32)
        l.tofile("rnn.bin")
        #plt.plot(losses)
        #plt.savefig("rnn2.jpg")

    def train_contrast(self):
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

#trainer.train_contrast()
trainer.train_one_fun()

#trainer.save_opti()
#optimizer_0 = grader(hidden_size,layers,batch_size,0,lr)
#optimizer_0.feed(tf.reshape(params[0][0][0],[1,1,1]))



