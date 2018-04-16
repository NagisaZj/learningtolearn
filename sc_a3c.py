import multiprocessing, threading, gym, os, shutil
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import pysc2
from pysc2 import agents,env
from pysc2.env import sc2_env
from pysc2.agents import base_agent
from pysc2 import lib
from pysc2.env import environment
from absl import flags ,app


MAX_GLOBAL_EP = 20000
GLOBAL_NET_SCOPE="Global_Net"
UPDATE_GLOBAL_ITER = 40
scr_pixels=84
scr_num=17
mini_pixels=64
mini_num=7
select_num=2
select_depth=7
scr_bound=[0,scr_pixels-1]
mini_bound=[0,mini_pixels-1]
entropy_gamma=0.005
steps=40
action_speed=8
reward_discount=GAMMA=0.9
LR_A = 0.00002    # learning rate for actor
LR_C = 0.0001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
N_WORKERS = 2
N_A=2
available_len = 524
available_len_used = 2


class ACnet(base_agent.BaseAgent):
    def __init__(self,scope,globalAC=None):
        super(ACnet, self).__init__()
        self.scope=scope
        if scope == GLOBAL_NET_SCOPE:  #build global net
            with tf.variable_scope(scope):
                self.s_scr=tf.placeholder(tf.float32,[None,scr_num,scr_pixels,scr_pixels],"S_scr")
                self.s_mini=tf.placeholder(tf.float32,[None,mini_num,mini_pixels,mini_pixels],"S_mini")
                self.s_select=tf.placeholder(tf.float32,[None,select_num*select_depth],"S_select")
                self.available=tf.placeholder(tf.float32,[None, available_len_used],"available")
                self._build_net()

                self.a_params = tl.layers.get_variables_with_name(scope + '/actor', True, False)
                self.c_params = tl.layers.get_variables_with_name(scope + '/critic', True, False)
                normal_dist_1=tf.contrib.distributions.Normal(self.mu_1,self.sigma_1)
                normal_dist_2=tf.contrib.distributions.Normal(self.mu_2,self.sigma_2)

                with tf.name_scope("choose_a"):  #choose actions,do not include a0 as a0 is discrete
                    self.mu_1, self.sigma_1 = self.mu_1 * scr_bound[1], self.sigma_1 + 1e-5
                    self.mu_2, self.sigma_2 = self.mu_2 * mini_bound[1], self.sigma_2 + 1e-5
                    self.a_1=tf.clip_by_value(tf.squeeze(normal_dist_1.sample(1),axis=0),*scr_bound)
                    self.a_2=tf.clip_by_value(tf.squeeze(normal_dist_2.sample(1),axis=0),*mini_bound)

        else:
            with tf.variable_scope(scope): #else, build local network
                self.s_scr = tf.placeholder(tf.float32, [None, scr_num, scr_pixels, scr_pixels], "S_scr")
                self.s_mini = tf.placeholder(tf.float32, [None, mini_num, mini_pixels, mini_pixels], "S_mini")
                self.s_select = tf.placeholder(tf.float32, [None, select_num , select_depth], "S_select")
                self.available = tf.placeholder(tf.float32, [None, available_len_used], "available")
                self.a0 = tf.placeholder(tf.int32,[None,1],"a0")
                self.a1 = tf.placeholder(tf.float32, [None, 1], "a1")
                self.a2 = tf.placeholder(tf.float32, [None, 1], "a2")
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                self._build_net()

                td=tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

            with tf.name_scope('wrap_a_out'):
                self.test = self.sigma_1[0]
                self.mu_1, self.sigma_1 = self.mu_1 * scr_bound[1], self.sigma_1 + 1e-5
                self.mu_2, self.sigma_2 = self.mu_2 * mini_bound[1], self.sigma_2 + 1e-5

            normal_dist_1 = tf.contrib.distributions.Normal(self.mu_1, self.sigma_1)
            normal_dist_2 = tf.contrib.distributions.Normal(self.mu_2, self.sigma_2)

            with tf.name_scope("a_loss"):    #build loss function
                log_prob0=tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a0, N_A, dtype=tf.float32), axis=1,
                              keep_dims=True)
                log_prob1 = normal_dist_1.log_prob(self.a1)
                log_prob2 = normal_dist_2.log_prob(self.a2)
                log_prob=tf.zeros_like(log_prob0)
                print(self.a0.shape)
                '''
                for i in range(self.a0.shape[0]):
                    if self.a0[i,0]!=0:
                        log_prob[i,0]=log_prob0[i,0]+log_prob1[i,0]+log_prob2[i,0]
                    else:
                        log_prob[i,0]=log_prob0[i,0]
                '''
                log_prob=log_prob0+log_prob1+log_prob2

                exp_v=log_prob*td

                entropy0=-tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)
                entropy1=normal_dist_1.entropy()
                entropy2=normal_dist_2.entropy()
                '''
                for i in range(self.a0.shape[0]):
                    if self.a0[i,0]!=0:
                        entropy[i,0] = entropy0[i,0] + entropy1[i,0] + entropy2[i,0]
                    else:
                        entropy[i, 0] = entropy0[i, 0]
                '''
                entropy=entropy0+entropy1+entropy2  #add entropy to encourage exploration

                # TODO: action a0(select all) and action a1(move_screen) should have different entropy and loss,
                # TODO: as the number of parameters are different(1 for a0, and 3 for a1) HOW TO IMPLEMENT?


                self.exp_v=entropy*entropy_gamma+exp_v
                self.a_loss=tf.reduce_mean(-self.exp_v)

            with tf.name_scope('choose_a'):  # use local params to choose action
                self.a_1 = tf.clip_by_value(tf.squeeze(normal_dist_1.sample(1), axis=0), *scr_bound)
                self.a_2 = tf.clip_by_value(tf.squeeze(normal_dist_2.sample(1), axis=0), *mini_bound)


            with tf.name_scope('local_grad'):
                self.a_params = tl.layers.get_variables_with_name(scope + '/actor', True, False)
                self.c_params = tl.layers.get_variables_with_name(scope + '/critic', True, False)
                self.a_grads = tf.gradients(self.a_loss, self.a_params)
                self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
        tl.layers.initialize_global_variables(sess)


    def update_global(self, feed_dict):  # run by a local
        _, _, t = sess.run([self.update_a_op, self.update_c_op, self.test], feed_dict)  # local grads applies to global net
        return t

    def pull_global(self):  # run by a local
        sess.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, scr, mini ,multi,avail_new):  # run by a local
        #scr = scr[np.newaxis, :]
        #mini = mini[np.newaxis, :]
        #multi = multi[np.newaxis, :]
        prob_weights = sess.run(self.a_prob, feed_dict={self.s_scr: scr,self.s_mini:mini,self.s_select:multi,
                                                        self.available:avail_new})
        a0 = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())
        #print(prob_weights)
        a1=sess.run([self.a_1], {self.s_scr: scr,self.s_mini:mini,self.s_select:multi})[0]
        a2 = sess.run([self.a_2], {self.s_scr: scr, self.s_mini: mini, self.s_select: multi})[0]
        #print(a1)
        return a0,a1,a2

    def save_ckpt(self):
        tl.files.exists_or_mkdir(self.scope)
        tl.files.save_ckpt(sess=sess, mode_name='model.ckpt', var_list=self.a_params+self.c_params, save_dir=self.scope, printable=True)

    def load_ckpt(self):
        tl.files.load_ckpt(sess=sess, var_list=self.a_params+self.c_params, save_dir=self.scope, printable=True)

    '''
    def step(self,timesteps):
        super(ACnet, self).step(timesteps)
    '''



    def _build_net(self):
        w_init = tf.contrib.layers.xavier_initializer()
        b_init = tf.truncated_normal_initializer(stddev=0.1)
        with tf.variable_scope("actor"):
            scr=InputLayer(self.s_scr,name="scr_in")
            scr=TransposeLayer(scr,[0,2,3,1],name="scr_t")
            scr=Conv2d(scr,n_filter=16,filter_size=(8,8),strides=(4,4),padding="VALID",
                       act=tf.nn.relu6,W_init=w_init,b_init=b_init,name="scr1")
            scr = Conv2d(scr, n_filter=32, filter_size=(4, 4), strides=(2, 2),padding="VALID",
                         act=tf.nn.relu6, W_init=w_init, b_init=b_init, name="scr2")
            scr=PoolLayer(scr, ksize=[1, scr.outputs.shape[1], scr.outputs.shape[2], 1], padding="VALID", pool=tf.nn.avg_pool,
                                   name="scr_GAP")
            scr = FlattenLayer(scr, name="scr_flat")

            mini=InputLayer(self.s_mini,name="mini_in")
            mini=TransposeLayer(mini,[0,2,3,1],name="mini_t")
            mini=Conv2d(mini,n_filter=16,filter_size=(8,8),strides=(4,4),padding="VALID",
                       act=tf.nn.relu6,W_init=w_init,b_init=b_init,name="mini1")
            mini = Conv2d(mini, n_filter=32, filter_size=(4, 4), strides=(2, 2),padding="VALID",
                         act=tf.nn.relu6, W_init=w_init, b_init=b_init, name="mini2")
            mini=PoolLayer(mini, ksize=[1, mini.outputs.shape[1], mini.outputs.shape[2], 1], padding="VALID", pool=tf.nn.avg_pool,
                                   name="mini_GAP")
            mini = FlattenLayer(mini, name="mini_flat")


            multi=InputLayer(self.s_select,name="select_in")
            multi=FlattenLayer(multi,name="flat")
            multi=DenseLayer(multi,10,act=tf.nn.tanh,W_init=w_init,b_init=b_init,name="multi_dense")
            info_all=ConcatLayer([scr,mini,multi],name="concat")

            info_all=DenseLayer(info_all,100,act=tf.nn.relu6,W_init=w_init,b_init=b_init,name="Relu")
            a_prob=DenseLayer(info_all,2,tf.nn.softmax,W_init=w_init,b_init=b_init,name="action_prob")
            #TODO:MASK a_prob WITH AVAILABLE  DONE!
            arg1_mu = DenseLayer(info_all,1,act=tf.nn.tanh,W_init=w_init,b_init=b_init,name="argu_1_mu")
            arg1_sigma = DenseLayer(info_all, 1,act=tf.nn.softplus, W_init=w_init, b_init=b_init, name="argu_1_sigma")
            arg2_mu = DenseLayer(info_all, 1, act=tf.nn.tanh, W_init=w_init, b_init=b_init, name="argu_2_mu")
            arg2_sigma = DenseLayer(info_all, 1, act=tf.nn.softplus, W_init=w_init, b_init=b_init, name="argu_2_sigma")
            self.a_prob=a_prob.outputs    # probability for a0
            self.a_prob = tf.multiply(self.a_prob,self.available)
            self.a_prob=self.a_prob+1e-5 # added to avoid dividing by zero
            self.a_prob=self.a_prob / tf.reduce_sum(self.a_prob,1,keep_dims=True)
            self.mu_1=arg1_mu.outputs
            self.sigma_1=arg1_sigma.outputs       # average and std for a1
            self.mu_2 = arg2_mu.outputs
            self.sigma_2 = arg2_sigma.outputs      # average and std for s2
            self.test1=scr

        with tf.variable_scope("critic"):
            scr = InputLayer(self.s_scr, name="scr_in")
            scr = TransposeLayer(scr, [0, 2, 3, 1], name="scr_t")
            scr = Conv2d(scr, n_filter=16, filter_size=(8, 8), strides=(4, 4), padding="VALID",
                         act=tf.nn.relu6, W_init=w_init, b_init=b_init, name="scr1")
            scr = Conv2d(scr, n_filter=32, filter_size=(4, 4), strides=(2, 2), padding="VALID",
                         act=tf.nn.relu6, W_init=w_init, b_init=b_init, name="scr2")
            scr = PoolLayer(scr, ksize=[1, scr.outputs.shape[1], scr.outputs.shape[2], 1], padding="VALID",
                            pool=tf.nn.avg_pool,
                            name="scr_GAP")
            scr = FlattenLayer(scr, name="scr_flat")

            mini = InputLayer(self.s_mini, name="mini_in")
            mini = TransposeLayer(mini, [0, 2, 3, 1], name="mini_t")
            mini = Conv2d(mini, n_filter=16, filter_size=(8, 8), strides=(4, 4), padding="VALID",
                          act=tf.nn.relu6, W_init=w_init, b_init=b_init, name="mini1")
            mini = Conv2d(mini, n_filter=32, filter_size=(4, 4), strides=(2, 2), padding="VALID",
                          act=tf.nn.relu6, W_init=w_init, b_init=b_init, name="mini2")
            mini = PoolLayer(mini, ksize=[1, mini.outputs.shape[1], mini.outputs.shape[2], 1], padding="VALID",
                             pool=tf.nn.avg_pool,
                             name="mini_GAP")
            mini = FlattenLayer(mini, name="mini_flat")

            multi = InputLayer(self.s_select, name="select_in")
            multi = FlattenLayer(multi, name="flat")
            multi = DenseLayer(multi, 10, act=tf.nn.tanh, W_init=w_init, b_init=b_init, name="multi_dense")
            info_all = ConcatLayer([scr, mini, multi], name="concat")

            info_all = DenseLayer(info_all, 100, act=tf.nn.relu6, W_init=w_init, b_init=b_init, name="Relu")
            v=DenseLayer(info_all,1,W_init=w_init, b_init=b_init, name="v")   #v is the value of state feeded
            self.v=v.outputs

        #sess.run(tf.global_variables_initializer())




class Worker(object):
    def __init__(self,name,globalAC):
        self.env=sc2_env.SC2Env(map_name="CollectMineralShards",screen_size_px=(scr_pixels,scr_pixels),
                                          minimap_size_px=(mini_pixels,mini_pixels),discount=reward_discount,
                                          visualize=False,step_mul=action_speed,game_steps_per_episode=steps*action_speed)  #init the env

        self.name=name
        self.AC=ACnet(name,globalAC)

    def pre_process(self,scr,mini,multi,available):
        scr_new=np.zeros_like(scr)
        mini_new=np.zeros_like(mini)
        avail_new = np.zeros([1,available_len_used],dtype=np.float32)
        avail_new[0][0] = 1 if 7 in available else 0
        avail_new [0][1] = 1 if 331 in available else 0
        for i in range(scr_num):
            scr_new[i]=scr[i]-np.mean(scr[i])
            scr_new[i]=scr_new[i]/(np.std(scr_new[i])+1e-5)    #preprocessing

            # TODO:this preprocess is not completely the same as Deepmind! HOW TO IMPROVE?

        for i in range(mini_num):
            mini_new[i]=mini[i]-np.mean(mini[i])
            mini_new[i]=mini_new[i]/(np.std(mini_new[i])+1e-5)    #preprocessing
        '''
        mini_new = mini - np.ones([7,64,64])*np.mean(mini, axis=(1, 2))
        mini_new = mini_new / (np.std(mini_new, axis=(1, 2)) + 1e-5)
        '''
        multi_new = np.log(multi+1)       #log to prevent large numbers

        scr_new=scr_new[np.newaxis, :]
        mini_new = mini_new[np.newaxis, :]
        multi_new = multi_new[np.newaxis, :]
        return scr_new,mini_new,multi_new,avail_new

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_scr, buffer_mini,buffer_multi, buffer_a0 ,buffer_a1, buffer_a2, buffer_r,buffer_avail = [], [],[], [],[],[],[],[]
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            timestep = self.env.reset()  #timestep[0] contains rewards, observations, etc. SEE pysc2 FOR MORE INFO
            ep_r=0
            while True:
                scr=timestep[0].observation["screen"]
                mini=timestep[0].observation["minimap"]
                multi=timestep[0].observation["multi_select"]
                available= timestep[0].observation["available_actions"]
                #print(331 in available)
                if(multi.shape[0]==0):
                    multi=np.zeros([2,7],dtype=np.float32)
                #print(multi.shape)
                scr_new,mini_new,multi_new,avail_new=self.pre_process(scr,mini,multi,available)
                a0,a1,a2=self.AC.choose_action(scr_new,mini_new,multi_new,avail_new)
                if a0 ==0:
                    act_fun=pysc2.lib.actions.FunctionCall(7, [[0]])
                else:
                    act_fun=pysc2.lib.actions.FunctionCall(331,[[0],[int(a1),int(a2)]])    #the action for env

                new_timestep=self.env.step([act_fun])   #push the game forward with action chosen
                scr_2 = new_timestep[0].observation["screen"]
                mini_2 = new_timestep[0].observation["minimap"]
                multi_2 = new_timestep[0].observation["multi_select"]
                available_2 = new_timestep[0].observation ["available_actions"]
                #print(multi_2)
                if (multi_2.shape[0] == 0):
                    multi_2 = np.zeros([ 2, 7], dtype=np.float32)
                scr_new_2, mini_new_2, multi_new_2,avail_new_2 = self.pre_process(scr_2, mini_2, multi_2,available_2)
                r=new_timestep[0].reward
                if r ==0:
                    r=-1
                done = (new_timestep[0].step_type == environment.StepType.LAST)
                #print(done)
                ep_r += r
                buffer_scr.append(scr_new)
                buffer_mini.append(mini_new)
                buffer_multi.append(multi_new)
                buffer_r.append(new_timestep[0].reward)
                buffer_a0.append(a0)
                buffer_a1.append(a1)
                buffer_a2.append(a2)
                buffer_avail.append(avail_new)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ = 0
                    else:
                        v_s_ = sess.run(self.AC.v, {self.AC.s_scr: scr_new_2,
                                                    self.AC.s_mini: mini_new_2,
                                                    self.AC.s_select: multi_new_2})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_          #compute v target
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_scr, buffer_mini, buffer_multi,buffer_a0,buffer_a1,buffer_a2,buffer_v_target ,buffer_avail= np.vstack(buffer_scr
                        ), np.vstack(buffer_mini), np.vstack(buffer_multi),np.vstack(buffer_a0),np.vstack(buffer_a1
                        ),np.vstack(buffer_a2), np.vstack(buffer_v_target)  , np.vstack(buffer_avail)   #put together into a single array
                    feed_dict = {
                        self.AC.s_scr: buffer_scr,
                        self.AC.s_mini: buffer_mini,
                        self.AC.s_select: buffer_multi,
                        self.AC.a0: buffer_a0,
                        self.AC.a1: buffer_a1,
                        self.AC.a2: buffer_a2,
                        self.AC.v_target: buffer_v_target,
                        self.AC.available : buffer_avail,
                    }

                    test = self.AC.update_global(feed_dict)  #update parameters
                    buffer_scr, buffer_mini, buffer_multi, buffer_a0, buffer_a1, buffer_a2, buffer_r,buffer_avail = [], [], [], [], [], [], [],[]

                    self.AC.pull_global()

                timestep=new_timestep
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.95 * GLOBAL_RUNNING_R[-1] + 0.05 * ep_r)
                    print(
                        self.name,
                        "episode:", GLOBAL_EP,
                        '| reward: %.1f' % ep_r,
                        "| running_reward: %.1f" % GLOBAL_RUNNING_R[-1],
                        # '| sigma:', test, # debug
                    )
                    GLOBAL_EP += 1
                    break





#a=ACnet("Global_Net")

def main(unused_argv):
    global sess
    global OPT_A, OPT_C
    global COORD
    global GLOBAL_AC
    sess = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')

        GLOBAL_AC = ACnet(GLOBAL_NET_SCOPE)  # we only need its params
        #tl.layers.initialize_global_variables(sess)
        #sess.run(tf.global_variables_initializer())

        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'Worker_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC))

    COORD = tf.train.Coordinator()

    tl.layers.initialize_global_variables(sess)
    #GLOBAL_AC.test1.print_params()
    #workers[0].AC.test1.print_params()

    ## start TF threading
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    GLOBAL_AC.save_ckpt()

if __name__=="__main__":

    app.run(main)








