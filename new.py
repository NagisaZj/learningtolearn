import multiprocessing, threading, gym, os, shutil
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
from optimizer import grader

GAME = 'BipedalWalker-v2' # BipedalWalkerHardcore-v2
OUTPUT_GRAPH = False
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
N_WORKERS = 1
MAX_GLOBAL_EP = 50000#8000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.999
ENTROPY_BETA = 0.005
LR_A = 0.00002    # learning rate for actor
LR_C = 0.0001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0    # will increase during training, stop training when it >= MAX_GLOBAL_EP
hidden_size = 20
mini_steps = 20
lr = 5e-3
p = 10
def sgn(v):
    return 1 if v>=0 else -1
env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.shape[0]
A_BOUND = [env.action_space.low, env.action_space.high]
# print(env.unwrapped.hull.position[0])
# exit()

class ACNet(object):
    def __init__(self, scope, globalAC=None):
        self.globalac = globalAC
        self.scope = scope
        if scope == GLOBAL_NET_SCOPE:
            ## global network only do inference
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                self._build_net()
                self.build_opti()
                self.apply_grads()
                self.update_opti()
                self.a_params = self.sd_a_params+self.tanh_params+self.sp_params
                self.c_params = self.sd_v_params+self.no_params

                normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma) # for continuous action space

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *A_BOUND)

        else:
            ## worker network calculate gradient locally, update on global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self._build_net()
                #self.build_opti()

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('wrap_a_out'):
                    self.test = self.sigma[0]
                    self.mu, self.sigma = self.mu * A_BOUND[1], self.sigma + 1e-5

                normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma) # for continuous action space

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), *A_BOUND)

                with tf.name_scope('local_grad'):
                    self.a_params = self.sd_a_params+self.tanh_params+self.sp_params
                    self.c_params = self.sd_v_params+self.no_params
                    self.sd_a_grads = tf.gradients(self.a_loss, self.sd_a_params)
                    self.tanh_grads = tf.gradients(self.a_loss, self.tanh_params)
                    self.sp_grads = tf.gradients(self.a_loss, self.sp_params)
                    self.sd_v_grads = tf.gradients(self.c_loss, self.sd_v_params)
                    self.no_grads = tf.gradients(self.c_loss, self.no_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params,globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                #with tf.name_scope('push'):
                #    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                #    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self):
        w_init = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('actor'):# Policy network
            self.W1 = tf.Variable(tf.random_uniform([N_S, 500], -1.0, 1.0), name='W1')
            self.b1 = tf.Variable(tf.zeros(shape=[500]), name='b1')
            self.y1 = tf.nn.relu6(tf.matmul(self.s, self.W1) + self.b1)
            self.W2 = tf.Variable(tf.random_uniform([500, 50], -1.0, 1.0), name='W2')
            self.b2 = tf.Variable(tf.zeros(shape=[50]), name='b2')
            self.y2 = tf.nn.relu6(tf.matmul(self.y1, self.W2) + self.b2)
            self.W3 = tf.Variable(tf.random_uniform([50, N_A], -1.0, 1.0), name='W3')
            self.b3 = tf.Variable(tf.zeros(shape=[N_A]), name='b3')
            self.mu = tf.nn.tanh(tf.matmul(self.y2, self.W3) + self.b3)
            self.W4 = tf.Variable(tf.random_uniform([50, N_A], -1.0, 1.0), name='W4')
            self.b4 = tf.Variable(tf.zeros(shape=[N_A]), name='b4')
            self.sigma = tf.nn.softplus(tf.matmul(self.y2, self.W4) + self.b4)



        with tf.variable_scope('critic'):       # we use Value-function here, but not Q-function.
            self.W5 = tf.Variable(tf.random_uniform([N_S, 500], -1.0, 1.0), name='W5')
            self.b5 = tf.Variable(tf.zeros(shape=[500]), name='b5')
            self.y3 = tf.nn.relu6(tf.matmul(self.s, self.W5) + self.b5)
            self.W6 = tf.Variable(tf.random_uniform([500, 50], -1.0, 1.0), name='W6')
            self.b6 = tf.Variable(tf.zeros(shape=[50]), name='b6')
            self.y4 = tf.nn.relu6(tf.matmul(self.y3, self.W6) + self.b6)
            self.W7 = tf.Variable(tf.random_uniform([50, N_A], -1.0, 1.0), name='W7')
            self.b7 = tf.Variable(tf.zeros(shape=[N_A]), name='b7')
            self.v = tf.matmul(self.y4, self.W7) + self.b7
        self.sd_a_params = [self.W1,self.b1,self.W2,self.b2]
        self.sd_v_params = [self.W5,self.b5,self.W6,self.b6]
        self.tanh_params = [self.W3,self.b3]
        self.sp_params = [self.W4,self.b4]
        self.no_params = [self.W7,self.b7]
        #self.build_opti()


    def build_opti(self):
        self.sigmoid_a_num = 0

        for param in self.sd_a_params:
            if len(param.shape) == 1:
                self.sigmoid_a_num = self.sigmoid_a_num + int(param.shape[0])
            elif len(param.shape) == 2:
                self.sigmoid_a_num = self.sigmoid_a_num + int(param.shape[0]) * int(param.shape[1])
        self.sigmoid_a_optimizer = grader(hidden_size, "sd_a", lr, sess, self.sigmoid_a_num)
        self.sigmoid_a_optimizer.load()
        # self.state_sigmoid = self.sigmoid_optimizer.cell.zero_state(self.sigmoid_num, tf.float32)
        # print(self.state_sigmoid)

        self.sigmoid_v_num = 0

        for param in self.sd_v_params:
            if len(param.shape) == 1:
                self.sigmoid_v_num = self.sigmoid_v_num + int(param.shape[0])
            elif len(param.shape) == 2:
                self.sigmoid_v_num = self.sigmoid_v_num + int(param.shape[0]) * int(param.shape[1])
        self.sigmoid_v_optimizer = grader(hidden_size, "sd_v", lr, sess, self.sigmoid_v_num)
        self.sigmoid_v_optimizer.load()

        self.tanh_num = 0

        for param in self.tanh_params:
            if len(param.shape) == 1:
                self.tanh_num = self.tanh_num + int(param.shape[0])
            elif len(param.shape) == 2:
                self.tanh_num = self.tanh_num + int(param.shape[0]) * int(param.shape[1])

        self.tanh_optimizer = grader(hidden_size, "tanh", lr, sess, self.tanh_num)
        self.tanh_optimizer.load()

        self.sp_num = 0

        for param in self.sp_params:
            if len(param.shape) == 1:
                self.sp_num = self.sp_num + int(param.shape[0])
            elif len(param.shape) == 2:
                self.sp_num = self.sp_num + int(param.shape[0]) * int(param.shape[1])

        self.sp_optimizer = grader(hidden_size, "sp", lr, sess, self.sp_num)
        self.sp_optimizer.load()

        self.no_num = 0

        for param in self.no_params:
            if len(param.shape) == 1:
                self.no_num = self.no_num + int(param.shape[0])
            elif len(param.shape) == 2:
                self.no_num = self.no_num + int(param.shape[0]) * int(param.shape[1])

        self.no_optimizer = grader(hidden_size, "no", lr, sess, self.no_num)
        self.no_optimizer.load()

    def apply_grads(self):
        #print(self.sigmoid_optimizer.output)
        self.update_sigmoid_a = self.sigmoid_a_optimizer.output
        self.update_tanh = self.tanh_optimizer.output
        self.update_sp = self.sp_optimizer.output
        self.update_sigmoid_v = self.sigmoid_v_optimizer.output
        self.update_no = self.no_optimizer.output
        self.update_sigmoid_a = tf.reshape(self.update_sigmoid_a,[self.update_sigmoid_a.shape[0]])
        self.update_tanh = tf.reshape(self.update_tanh, [self.update_tanh.shape[0]])
        self.update_sp = tf.reshape(self.update_sp, [self.update_sp.shape[0]])
        self.update_sigmoid_v = tf.reshape(self.update_sigmoid_v, [self.update_sigmoid_v.shape[0]])
        self.update_no = tf.reshape(self.update_no, [self.update_no.shape[0]])
        self.grads_new_sd_a = []
        self.grads_new_tanh = []
        self.grads_new_sp = []
        self.grads_new_sd_v = []
        self.grads_new_no = []

        num = 0

        for param in self.sd_a_params:

            if len(param.shape) == 1:
                params = tf.add(param , self.update_sigmoid_a[num:num + int(param.shape[0])])
                self.grads_new_sd_a.append(params)
                num = num + int(param.shape[0])


            elif len(param.shape) == 2:
                params = tf.add(param , tf.transpose(tf.reshape(
                        self.update_sigmoid_a[num: num + int(param.shape[0]) * int(param.shape[1])],
                        [param.shape[1], param.shape[0]])))
                self.grads_new_sd_a.append(params)
                num = num + int(param.shape[0])*int(param.shape[1])

        num = 0

        for param in self.tanh_params:

            if len(param.shape) == 1:
                params = tf.add(param , self.update_tanh[num:num + int(param.shape[0])])
                self.grads_new_tanh.append(params)
                num = num + int(param.shape[0])


            elif len(param.shape) == 2:
                params = tf.add(param , tf.transpose(tf.reshape(
                        self.update_tanh[num: num + int(param.shape[0]) * int(param.shape[1])],
                        [param.shape[1], param.shape[0]])))
                self.grads_new_tanh.append(params)
                num = num + int(param.shape[0])*int(param.shape[1])

        num = 0

        for param in self.sp_params:

            if len(param.shape) == 1:
                params = tf.add(param, self.update_sp[num:num + int(param.shape[0])])
                self.grads_new_sp.append(params)
                num = num + int(param.shape[0])


            elif len(param.shape) == 2:
                params = tf.add(param, tf.transpose(tf.reshape(
                    self.update_sp[num: num + int(param.shape[0]) * int(param.shape[1])],
                    [param.shape[1], param.shape[0]])))
                self.grads_new_sp.append(params)
                num = num + int(param.shape[0]) * int(param.shape[1])

        num = 0

        for param in self.sd_v_params:

            if len(param.shape) == 1:
                params = tf.add(param, self.update_sigmoid_v[num:num + int(param.shape[0])])
                self.grads_new_sd_v.append(params)
                num = num + int(param.shape[0])


            elif len(param.shape) == 2:
                params = tf.add(param, tf.transpose(tf.reshape(
                    self.update_sigmoid_v[num: num + int(param.shape[0]) * int(param.shape[1])],
                    [param.shape[1], param.shape[0]])))
                self.grads_new_sd_v.append(params)
                num = num + int(param.shape[0]) * int(param.shape[1])

        num = 0

        for param in self.no_params:

            if len(param.shape) == 1:
                params = tf.add(param, self.update_no[num:num + int(param.shape[0])])
                self.grads_new_no.append(params)
                num = num + int(param.shape[0])


            elif len(param.shape) == 2:
                params = tf.add(param, tf.transpose(tf.reshape(
                    self.update_no[num: num + int(param.shape[0]) * int(param.shape[1])],
                    [param.shape[1], param.shape[0]])))
                self.grads_new_no.append(params)
                num = num + int(param.shape[0]) * int(param.shape[1])

        self.y1_new = tf.nn.relu6(tf.matmul(self.s, self.grads_new_sd_a[0]) + self.grads_new_sd_a[1])
        self.y2_new = tf.nn.relu6(tf.matmul(self.y1_new, self.grads_new_sd_a[2]) + self.grads_new_sd_a[3])
        self.y3_new = tf.nn.relu6(tf.matmul(self.s, self.grads_new_sd_v[0]) + self.grads_new_sd_v[1])
        self.y4_new = tf.nn.relu6(tf.matmul(self.y3_new, self.grads_new_sd_v[2]) + self.grads_new_sd_v[3])
        self.mu_new = tf.nn.tanh(tf.matmul(self.y2_new, self.grads_new_tanh[0]) + self.grads_new_tanh[1])
        self.sigma_new = tf.nn.softplus(tf.matmul(self.y2_new, self.grads_new_sp[0]) + self.grads_new_sp[1])
        self.v_new = tf.matmul(self.y4_new, self.grads_new_no[0]) + self.grads_new_no[1]
        td = tf.subtract(self.v_target, self.v_new, name='TD_error')
        with tf.name_scope('c_loss'):
            self.c_loss_new = tf.reduce_mean(tf.square(td))

        with tf.name_scope('wrap_a_out'):
            #self.test = self.sigma[0]
            self.mu_new, self.sigma_new = self.mu_new * A_BOUND[1], self.sigma_new + 1e-5

        normal_dist = tf.contrib.distributions.Normal(self.mu_new, self.sigma_new)  # for continuous action space

        with tf.name_scope('a_loss'):
            log_prob = normal_dist.log_prob(self.a_his)
            exp_v = log_prob * td
            entropy = normal_dist.entropy()  # encourage exploration
            self.exp_v_new = ENTROPY_BETA * entropy + exp_v
            self.a_loss_new = tf.reduce_mean(-self.exp_v_new)
        self.assign_op = []
        self.assign_op.append(self.W1.assign(self.grads_new_sd_a[0]))
        self.assign_op.append(self.b1.assign(self.grads_new_sd_a[1]))
        self.assign_op.append(self.W2.assign(self.grads_new_sd_a[2]))
        self.assign_op.append(self.b2.assign(self.grads_new_sd_a[3]))
        self.assign_op.append(self.W3.assign(self.grads_new_tanh[0]))
        self.assign_op.append(self.b3.assign(self.grads_new_tanh[1]))
        self.assign_op.append(self.W4.assign(self.grads_new_sp[0]))
        self.assign_op.append(self.b4.assign(self.grads_new_sp[1]))
        self.assign_op.append(self.W5.assign(self.grads_new_sd_v[0]))
        self.assign_op.append(self.b5.assign(self.grads_new_sd_v[1]))
        self.assign_op.append(self.W6.assign(self.grads_new_sd_v[2]))
        self.assign_op.append(self.b6.assign(self.grads_new_sd_v[3]))
        self.assign_op.append(self.W7.assign(self.grads_new_no[0]))
        self.assign_op.append(self.b7.assign(self.grads_new_no[1]))

    def update_opti(self):
        self.sigmoid_a_optimizer.train(self.a_loss_new)
        self.tanh_optimizer.train(self.a_loss_new)
        self.sp_optimizer.train(self.a_loss_new)
        self.sigmoid_v_optimizer.train(self.c_loss_new)
        self.no_optimizer.train(self.c_loss_new)

    def save_opti(self):
        self.sigmoid_a_optimizer.save()
        self.tanh_optimizer.save()
        self.sp_a_optimizer.save()
        self.sigmoid_v_optimizer.save()
        self.no_optimizer.save()

    def update_global(self, feed_dict):  # run by a local
        _, _, t = sess.run([self.update_a_op, self.update_c_op, self.test], feed_dict)  # local grads applies to global net
        return t

    def pull_global(self):  # run by a local
        sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        s = s[np.newaxis, :]
        return sess.run(self.A, {self.s: s})[0]

    def save_ckpt(self):
        tl.files.exists_or_mkdir(self.scope)
        tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', var_list=self.a_params+self.c_params, save_dir=self.scope, printable=True)

    def load_ckpt(self):
        tl.files.load_ckpt(sess=sess, var_list=self.a_params+self.c_params, save_dir=self.scope, printable=True)
        # tl.files.load_ckpt(sess=sess, mode_name='model.ckpt', var_list=self.a_params+self.c_params, save_dir=self.scope, is_latest=False, printable=True)

class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME)
        self.name = name
        self.AC = ACNet(name, globalAC)

    def preprocess(self):
        sd_a_over = None
        tanh_over = None
        sp_over = None
        sd_v_over = None
        no_over = None
        for grad in self.sd_a:
            if len(grad.shape) == 1:
                sd_a_over = grad if sd_a_over is None else np.concatenate((sd_a_over, grad), axis=0)

            elif len(grad.shape) == 2:
                for j in range(grad.shape[1]):
                    sd_a_over = grad[:, j] if sd_a_over is None else np.concatenate((sd_a_over, grad[:, j]),
                                                                                          axis=0)

        for grad in self.tanh:
            if len(grad.shape) == 1:
                tanh_over = grad if tanh_over is None else np.concatenate((tanh_over, grad), axis=0)

            elif len(grad.shape) == 2:
                for j in range(grad.shape[1]):
                    tanh_over = grad[:, j] if tanh_over is None else np.concatenate((tanh_over, grad[:, j]),
                                                                                          axis=0)
        for grad in self.sp:
            if len(grad.shape) == 1:
                sp_over = grad if sp_over is None else np.concatenate((sp_over, grad), axis=0)

            elif len(grad.shape) == 2:
                for j in range(grad.shape[1]):
                    sp_over = grad[:, j] if sp_over is None else np.concatenate((sp_over, grad[:, j]),
                                                                                            axis=0)

        for grad in self.sd_v:
            if len(grad.shape) == 1:
                sd_v_over = grad if sd_v_over is None else np.concatenate((sd_v_over, grad), axis=0)

            elif len(grad.shape) == 2:
                for j in range(grad.shape[1]):
                    sd_v_over = grad[:, j] if sd_v_over is None else np.concatenate((sd_v_over, grad[:, j]),
                                                                                            axis=0)
        for grad in self.no:
            if len(grad.shape) == 1:
                no_over = grad if no_over is None else np.concatenate((no_over, grad), axis=0)

            elif len(grad.shape) == 2:
                for j in range(grad.shape[1]):
                    no_over = grad[:, j] if no_over is None else np.concatenate((no_over, grad[:, j]),
                                                                                            axis=0)
        # self.softmax_over = softmax_over.reshape((softmax_over.shape[0], 1, 1))
        sd_a_true = np.zeros((sd_a_over.shape[0], 1, 2), dtype=np.float32)
        tanh_true = np.zeros((tanh_over.shape[0], 1, 2), dtype=np.float32)
        sp_true = np.zeros((sp_over.shape[0], 1, 2), dtype=np.float32)
        sd_v_true = np.zeros((sd_v_over.shape[0], 1, 2), dtype=np.float32)
        no_true = np.zeros((no_over.shape[0], 1, 2), dtype=np.float32)
        for i in range(sd_a_over.shape[0]):
            sd_a_true[i, 0, 0] = np.log(abs(sd_a_over[i])) / p if abs(sd_a_over[i]) >= np.power(np.e,
                                                                                                    p * -1) else -1
            sd_a_true[i, 0, 1] = sgn(sd_a_over[i]) if abs(sd_a_over[i]) >= np.power(np.e, p * -1) \
                else sd_a_over[i] * np.power(np.e, p)

        for i in range(tanh_over.shape[0]):
            tanh_true[i, 0, 0] = np.log(abs(tanh_over[i])) / p if abs(tanh_over[i]) >= np.power(np.e,
                                                                                                    p * -1) else -1
            tanh_true[i, 0, 1] = sgn(tanh_over[i]) if abs(tanh_over[i]) >= np.power(np.e, p * -1) \
                else tanh_over[i] * np.power(np.e, p)
        for i in range(sp_over.shape[0]):
            sp_true[i, 0, 0] = np.log(abs(sp_over[i])) / p if abs(sp_over[i]) >= np.power(np.e,
                                                                                                        p * -1) else -1
            sp_true[i, 0, 1] = sgn(sp_over[i]) if abs(sp_over[i]) >= np.power(np.e, p * -1) \
                else sp_over[i] * np.power(np.e, p)

        for i in range(sd_v_over.shape[0]):
            sd_v_true[i, 0, 0] = np.log(abs(sd_v_over[i])) / p if abs(sd_v_over[i]) >= np.power(np.e,
                                                                                                        p * -1) else -1
            sd_v_true[i, 0, 1] = sgn(sd_v_over[i]) if abs(sd_v_over[i]) >= np.power(np.e, p * -1) \
                else sd_v_over[i] * np.power(np.e, p)

        for i in range(no_over.shape[0]):
            no_true[i, 0, 0] = np.log(abs(no_over[i])) / p if abs(no_over[i]) >= np.power(np.e,
                                                                                                        p * -1) else -1
            no_true[i, 0, 1] = sgn(no_over[i]) if abs(no_over[i]) >= np.power(np.e, p * -1) \
                else no_over[i] * np.power(np.e, p)

        self.sd_a = sd_a_true
        self.tanh = tanh_true
        self.sp = sp_true
        self.sd_v = sd_v_true
        self.no = no_true
    def grad_build(self):
        self.sd_a_all = np.zeros((self.sd_a.shape[0],mini_steps),dtype = np.float32)
        self.tanh_all = np.zeros((self.tanh.shape[0], mini_steps),dtype = np.float32)
        self.sp_all = np.zeros((self.sp.shape[0], mini_steps),dtype = np.float32)
        self.sd_v_all = np.zeros((self.sd_v.shape[0], mini_steps),dtype = np.float32)
        self.no_all = np.zeros((self.no.shape[0], mini_steps),dtype = np.float32)
        self.sd_a_all[:,0] = self.sd_a
        self.tanh_all[:, 0] = self.tanh
        self.sp_all[:, 0] = self.sp
        self.sd_v_all[:, 0] = self.sd_v
        self.no_all[:, 0] = self.no
    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        grad_count = 0
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            while True:
                ## visualize Worker_0 during training
                #if self.name == 'Worker_0':# and total_step % 30 == 0:
                    #self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)

                ## set robot falls reward to -2 instead of -100
                if r == -100: r = -2

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.sd_a,self.tanh,self.sp,self.sd_v,self.no = sess.run([self.AC.sd_a_grads,self.AC.tanh_grads,
                                                     self.AC.sp_grads,self.AC.sd_v_grads,self.AC.no_grads],
                                                    feed_dict = feed_dict)
                    self.preprocess()
                    if grad_count ==0:
                        self.grad_build()
                        grad_count = grad_count+1
                    else:
                        self.sd_a_all[:, grad_count] = self.sd_a
                        self.tanh_all[:, grad_count] = self.tanh
                        self.sp_all[:, grad_count] = self.sp
                        self.sd_v_all[:, grad_count] = self.sd_v
                        self.no_all[:, grad_count] = self.no
                        grad_count = grad_count + 1

                    feed_opti = {
                        self.AC.globalac.s: buffer_s,
                        self.AC.globalac.a_his: buffer_a,
                        self.AC.globalac.v_target: buffer_v_target,
                        self.AC.globalac.sigmoid_a_optimizer.input: self.sd_a,
                        self.AC.globalac.tanh_optimizer.input: self.tanh,
                        self.AC.globalac.sp_optimizer.input: self.sp,
                        self.AC.globalac.sigmoid_v_optimizer.input: self.sd_v,
                        self.AC.globalac.no_optimizer.input: self.no,
                    }
                    ## update gradients on global network
                    if grad_count ==20:
                        feed_opti_all = {
                        self.AC.globalac.s: buffer_s,
                        self.AC.globalac.a_his: buffer_a,
                        self.AC.globalac.v_target: buffer_v_target,
                        self.AC.globalac.sigmoid_a_optimizer.input:self.sd_a_all,
                        self.AC.globalac.tanh_optimizer.input: self.tanh_all,
                        self.AC.globalac.sp_optimizer.input: self.sp_all,
                        self.AC.globalac.sigmoid_v_optimizer.input: self.sd_v_all,
                        self.AC.globalac.no_optimizer.input: self.no_all,
                    }
                    #for j in range(mini_steps):
                        sess.run([self.AC.globalac.sigmoid_a_optimizer.train_op,
                                  self.AC.globalac.tanh_optimizer.train_op,
                                  self.AC.globalac.sp_optimizer.train_op,
                                  self.AC.globalac.sigmoid_v_optimizer.train_op,
                                  self.AC.globalac.no_optimizer.train_op],
                                      feed_dict=feed_opti_all)
                        grad_count = 0
                    sess.run([self.AC.globalac.assign_op], feed_dict=feed_opti)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    ## update local network from global network
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)
                    print(
                        self.name,
                        "episode:", GLOBAL_EP,
                        "| pos: %i" % self.env.unwrapped.hull.position[0],  # number of move
                        '| reward: %.1f' % ep_r,
                        "| running_reward: %.1f" % GLOBAL_RUNNING_R[-1],
                        # '| sigma:', test, # debug
                        'WIN '*5 if self.env.unwrapped.hull.position[0] >= 88 else '',
                    )
                    GLOBAL_EP += 1
                    #if GLOBAL_EP % 1000 ==0:
                    #    self.AC.save_ckpt()
                    break

def test():
    #print(N_A)
    ac = ACNet(GLOBAL_NET_SCOPE)
    env = gym.make(GAME)
    ac.load_ckpt()
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = ac.choose_action(s)
        s, r, done, info = env.step(a)
        if r == -100 : r = -2
        ep_r += r
        if done:
            s = env.reset()
            print(ep_r)
            ep_r = 0




if __name__ == "__main__":
    sess = tf.Session()
    #test()
    a = 1

    ###============================= TRAINING ===============================###
    #with  tf.device("/cpu:0"):
    OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
    OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
    GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params4
    #GLOBAL_AC.load_ckpt()
    workers = []
    # Create worker
    for i in range(N_WORKERS):
        i_name = 'Worker_%i' % i   # worker name
        workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()
    tl.layers.initialize_global_variables(sess)

    ## start TF threading
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
        #GLOBAL_AC.save_ckpt()
    COORD.join(worker_threads)

    GLOBAL_AC.save_ckpt()
    reward = np.array(GLOBAL_RUNNING_R, dtype=np.float32)
    reward.tofile("aa.bin")
