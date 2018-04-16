import numpy as np
import tensorflow as tf
import tensorlayer as tl
from optimizer import grader
import matplotlib.pyplot as plt

n_dimension = 20
net_size = 10
hidden_size=5
layers = 2
batch_size = 1
lr = 1e-3
full_batch = 100
train_steps = 10000
sess = tf.Session()



class train:
    def __init__(self,sess):
        self.optimizers = []
        self.sess = sess
        self.grad_list = []
        self.params = []
        return
    def build_target_net(self,times):
        with tf.variable_scope("optimizee%d"%times) as scope:
            w_init = tf.contrib.layers.xavier_initializer()
            self.input = tf.placeholder(tf.float32,[None,n_dimension])
            net = tl.layers.InputLayer(self.input,"In")
            net = tl.layers.DenseLayer(net,n_units=net_size,act = tf.nn.sigmoid,W_init = w_init, name="dense1")
            net = tl.layers.DenseLayer(net,n_units=1,act =tf.nn.sigmoid,W_init = w_init, name="dense2")
            output = net.outputs
            self.label = tf.placeholder(tf.float32,[None,1])
            loss = tf.reduce_mean(tf.square(output-self.label))
            self.loss = loss
            self.params = net.all_params
            self.gradients = tf.gradients(loss,self.params)
            self.optimizer = tf.train.AdamOptimizer(1e-3)
            self.loss_op = self.optimizer.minimize(self.loss)



    def build_opti(self):
        num = 0
        for param in self.params:
            if len(param.shape) ==1:
                for i in range(param.shape[0]):
                    gra = grader(hidden_size, layers, batch_size, num, lr, self.sess)
                    gra.load()
                    self.optimizers.append(gra)
                    num = num + 1

            elif len(param.shape) ==2:
                for i in range(param.shape[0]):
                    for j in range(param.shape[1]):
                        gra = grader(hidden_size, layers, batch_size, num, lr, self.sess)
                        gra.load()
                        self.optimizers.append(gra)
                        num = num + 1

    def out_grads(self):
        num = 0
        for grad in self.gradients:
            if len(grad.shape) == 1 :
                for i in range(grad.shape[0]):
                    gra =self.optimizers[num].feed(tf.reshape(grad[i],[1,1,1]))
                    self.grad_list.append(gra)
                    num = num + 1

            elif len(grad.shape) == 2 :
                for i in range(grad.shape[0]):
                    for j in range(grad.shape[1]):
                        gra =self.optimizers[num].feed(tf.reshape(grad[i,j],[1,1,1]))
                        self.grad_list.append(gra)
                        num = num + 1

    def apply_grads(self):
        num = 0
        for param in self.params:
            if len(param.shape) == 1 :
                for i in range(param.shape[0]):
                    tf.assign(param[i] , param[i]+self.grad_list[num])
                    num = num + 1

            elif len(param.shape) == 2 :
                for i in range(param.shape[0]):
                    tf.assign(param[i], param[i]+tf.stack(self.grad_list[ num : num + int(param.shape[1]) ]))
                    num = num + int(param.shape[1])


        self.grad_list = []

    def update(self):
        for opt in self.optimizers:
            opt.train(self.loss)


    def save_opti(self):
        for opt in self.optimizers:
            opt.save()


    def train_one_fun(self):
        W = np.random.rand(full_batch,n_dimension)
        y = np.random.rand(full_batch,1)
        feed_dict = {self.input:W,self.label:y}
        losses = []
        for i in range(train_steps):
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
        W = np.random.rand(full_batch, n_dimension)
        y = np.random.rand(full_batch, 1)
        losses = []
        feed_dict = {self.input: W, self.label: y}
        for i in range(train_steps):
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
trainer.build_target_net(0)
trainer.build_opti()
trainer.out_grads()
trainer.apply_grads()
trainer.update()
tl.layers.initialize_global_variables(sess)
trainer.train_contrast()
trainer.train_one_fun()

#trainer.save_opti()
#optimizer_0 = grader(hidden_size,layers,batch_size,0,lr)
#optimizer_0.feed(tf.reshape(params[0][0][0],[1,1,1]))



