import numpy as np
import pysc2
from pysc2 import agents
from pysc2.agents import base_agent
from pysc2 import lib


import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
import random

def read_data_simple():
    data_size=990
    screen=np.fromfile("C:/Users/probe/project_sc/sc/data/screen.bin",dtype=np.int32)
    screen_next=np.fromfile("C:/Users/probe/project_sc/sc/data/screen_next.bin",dtype=np.int32)
    minimap=np.fromfile("C:/Users/probe/project_sc/sc/data/minimap.bin",dtype=np.int32)
    minimap_next = np.fromfile("C:/Users/probe/project_sc/sc/data/mini_next.bin", dtype=np.int32)
    player=np.fromfile("C:/Users/probe/project_sc/sc/data/player.bin",dtype=np.int32)
    multi_select = np.fromfile("C:/Users/probe/project_sc/sc/data/multi_select.bin", dtype=np.int32)
    multi_select_next=np.fromfile("C:/Users/probe/project_sc/sc/data/multi_select_next.bin", dtype=np.int32)
    actions=np.fromfile("C:/Users/probe/project_sc/sc/data/actions.bin",dtype=np.int32)
    action_args=np.fromfile("C:/Users/probe/project_sc/sc/data/action_args.bin",dtype=np.int32)

    screen = screen.reshape(data_size, 17, 84, 84)
    screen = screen.astype(np.float32)
    screen_next = screen_next.reshape(data_size, 17, 84, 84)
    screen_next = screen_next.astype(np.float32)
    minimap = minimap.reshape(data_size, 7, 64, 64)
    minimap = minimap.astype(np.float32)
    minimap_next = minimap_next.reshape(data_size, 7, 64, 64)
    minimap_next = minimap_next.astype(np.float32)
    player = player.reshape(data_size, 11)
    player = player.astype(np.float32)
    multi_select = multi_select.reshape(data_size, 128, 7)
    multi_select = multi_select.astype(np.float32)
    multi_select_next = multi_select_next.reshape(data_size, 128, 7)
    multi_select_next = multi_select_next.astype(np.float32)
    multi_select = multi_select[:, 0:2, :]
    multi_select_next = multi_select_next[:, 0:2, :]
    action_args = action_args.reshape(data_size, 4)
    return screen,minimap,multi_select,actions,action_args,screen_next,minimap_next,multi_select_next,player

class data:
    def __init__(self):
        self.data_size=990
        self.epsilon=1e-7
        self.screen,self.minimap,self.multi_select,self.actions,self.action_args,self.screen_next,self.minimap_next,self.multi_select_next,self.player=read_data_simple()
        self.max_data_size=5000
        self.data_pointer=990
        self.next_data_pointer=990
        self.data_mark=990
        self.next_data_mark=990
        self.guiyihua()
        self.data_process()
        self.score_rem=[]




    def guiyihua(self):
            for i in range(self.data_size):
                for j in range(17):
                    self.screen[i][j]=self.screen[i][j]-np.ones((84,84))*self.screen[i][j].mean()
                    self.screen[i][j]=self.screen[i][j]/(self.screen[i][j].max()+self.epsilon)
                    self.screen_next[i][j] = self.screen_next[i][j] - np.ones((84, 84)) *self.screen_next[i][j].mean()
                    self.screen_next[i][j] = self.screen_next[i][j] / (self.screen_next[i][j].max()+self.epsilon)
                    #print(self.screen[i][j].max())
                for j in range(7):
                    self.minimap[i][j] = self.minimap[i][j] - np.ones((64, 64)) * self.minimap[i][j].mean()
                    self.minimap[i][j] = self.minimap[i][j] / (self.minimap[i][j].max()+self.epsilon)
                    self.minimap_next[i][j] = self.minimap_next[i][j] - np.ones((64, 64)) * self.minimap_next[i][j].mean()
                    self.minimap_next[i][j] = self.minimap_next[i][j] / (self.minimap_next[i][j].max()+self.epsilon)
                    self.multi_select[i,:,j] = self.multi_select[i,:,j] - np.ones((2)) * self.multi_select[i,:,j].mean()
                    self.multi_select[i,:,j] = self.multi_select[i,:,j] / (self.multi_select[i,:,j].max()+self.epsilon)
                    self.multi_select_next[i,:,j] = self.multi_select_next[i,:,j] - np.ones((2)) * self.multi_select_next[i,:,j].mean()
                    self.multi_select_next[i,:,j] = self.multi_select_next[i,:,j] / (self.multi_select_next[i,:,j].max()+self.epsilon)


    def data_process(self):
        self.screen = self.screen.transpose((0, 2, 3, 1))
        self.minimap = self.minimap.transpose((0, 2, 3, 1))
        self.screen_next = self.screen_next.transpose((0, 2, 3, 1))
        self.minimap_next = self.minimap_next.transpose((0, 2, 3, 1))
        self.action_actual=np.zeros((self.max_data_size,7057),dtype=np.float32)
        for i in range(self.data_size):
            if(self.actions[i]==7):
                self.action_actual[i][7056]=1
            else:
                self.action_actual[i][self.action_args[i][0]*84+self.action_args[i][1]]=1

        self.score=np.zeros((self.max_data_size),dtype=np.int32)
        for i in range(self.data_size):
            if i>0:
                self.score[i]=(self.player[i,1]-self.player[i-1,1])/100 if self.player[i,1]>self.player[i-1,1] else 0
                #print(self.score[i])

        self.action_simple=np.zeros((self.max_data_size),dtype=np.float32)
        for i in range(self.data_size):
            if(self.actions[i]==7):
                self.action_simple[i]=7056
            else:
                self.action_simple[i]=self.action_args[i][0]*84+self.action_args[i][1]
            #print(self.action_simple[i])

        self.screen=np.concatenate((self.screen,np.zeros((self.max_data_size-self.data_size,84,84,17),dtype=np.float32)),axis=0)
        self.minimap=np.concatenate((self.minimap,np.zeros((self.max_data_size-self.data_size,64,64,7),dtype=np.float32)),axis=0)
        self.multi_select=np.concatenate((self.multi_select,np.zeros((self.max_data_size-self.data_size,2,7),dtype=np.float32)),axis=0)
        self.screen_next = np.concatenate((self.screen_next, np.zeros((self.max_data_size - self.data_size, 84, 84, 17),dtype=np.float32)), axis=0)
        self.minimap_next = np.concatenate(
            (self.minimap_next, np.zeros((self.max_data_size - self.data_size, 64, 64, 7),dtype=np.float32)), axis=0)
        self.multi_select_next = np.concatenate(
            (self.multi_select_next, np.zeros((self.max_data_size - self.data_size, 2, 7),dtype=np.float32)), axis=0)
        self.player=np.concatenate((self.player,np.zeros((self.max_data_size-self.data_size,11),dtype=np.float32)),axis=0)



    def data_add_first(self,scr,minimap,multi_select,player,act):
        print(act)
        action_actual = np.zeros((1, 7057), dtype=np.int32)
        action_actual[0][act] = 1
        if(self.data_pointer<self.max_data_size):
            self.screen[self.data_pointer]=scr
            self.minimap[self.data_pointer] = minimap
            self.multi_select[self.data_pointer] = multi_select
            self.action_simple[self.data_pointer] = np.array([act])
            self.player[self.data_pointer] = player
            self.score[self.data_pointer] = np.array([0])
            self.action_actual[self.data_pointer]=action_actual
            self.data_pointer=self.data_pointer+1

        else:
            self.screen[self.data_mark]=scr
            self.minimap[self.data_mark]=minimap
            self.multi_select[self.data_mark] = multi_select
            self.action_simple[self.data_mark]=np.array([act])
            self.score[self.data_mark]=np.array([0])
            self.player[self.data_mark]=player
            self.action_actual[self.data_mark]=action_actual
            self.data_mark=self.data_mark+1
            if(self.data_mark==self.max_data_size):
                self.data_mark=990


    def data_add_middle(self,scr,minimap,multi_select,player,act):
        action_actual = np.zeros((1, 7057), dtype=np.int32)
        action_actual[0][act] = 1
        if (self.data_pointer < self.max_data_size):
            self.screen[self.data_pointer] = scr
            self.minimap[self.data_pointer] = minimap
            self.multi_select[self.data_pointer] = multi_select
            self.action_simple[self.data_pointer] = np.array([act])
            self.player[self.data_pointer] = player
            self.score[self.data_pointer] =np.array([(player[0,1]-self.player[self.data_pointer-1,1])/100 if player[0,1]>self.player[self.data_pointer-1,1] else 0])
            self.action_actual[self.data_pointer] = action_actual
            self.data_pointer = self.data_pointer + 1

        else:
            self.screen[self.data_mark] = scr
            self.minimap[self.data_mark] = minimap
            self.multi_select[self.data_mark] = multi_select
            self.action_simple[self.data_mark] = np.array([act])
            self.score[self.data_mark] = np.array([(player[0,1]-self.player[self.data_pointer-1,1])/100 if player[0,1]>self.player[self.data_pointer-1,1] else 0])
            self.player[self.data_mark] = player
            self.action_actual[self.data_mark] = action_actual
            self.data_mark = self.data_mark + 1
            if (self.data_mark == self.max_data_size):
                self.data_mark = 990



        if self.next_data_pointer<self.max_data_size :
            self.screen_next[self.next_data_pointer] = scr
            self.minimap_next[self.next_data_pointer] =minimap
            self.multi_select_next[self.next_data_pointer] = multi_select
            self.next_data_pointer=self.next_data_pointer+1
        else:
            self.screen_next[self.next_data_mark] = scr
            self.minimap_next[self.next_data_mark] = minimap
            self.multi_select_next[self.next_data_mark] = multi_select
            self.next_data_mark = self.next_data_mark + 1
            if (self.next_data_mark == self.max_data_size):
                self.next_data_mark = 990




    def data_add_end(self,scr,minimap,multi_select,player,act):
        if self.next_data_pointer<self.max_data_size :
            self.screen_next[self.next_data_pointer] = scr
            self.minimap_next[self.next_data_pointer] =minimap
            self.multi_select_next[self.next_data_pointer] = multi_select
            self.next_data_pointer=self.next_data_pointer+1
        else:
            self.screen_next[self.next_data_mark] = scr
            self.minimap_next[self.next_data_mark] = minimap
            self.multi_select_next[self.next_data_mark] = multi_select
            self.next_data_mark = self.next_data_mark + 1
            if (self.next_data_mark == self.max_data_size):
                self.next_data_mark = 990
        self.score_rem.append(player[0,1])
        self.outarr=np.array(self.score_rem)
        np.savetxt("final_score.txt", self.outarr)
    #def sample(self,batch_size):


class dqfd_less(base_agent.BaseAgent):
    def __init__(self):
        super(dqfd_less, self).__init__()
        self.reward = 0
        self.episodes = 0
        self.steps = 0
        self.obs_spec = None
        self.action_spec = None
        self.data_size = 990
        self.timestep=0
        self.batchsize=64
        self.gamma=0.7
        self.greedy=0.7
        self.record=0
        self.lr=1e-4
        self.data=data()
        self.screen_in = tf.placeholder(tf.float32, shape=[None, 84, 84, 17], name="screen")
        self.minimap_in = tf.placeholder(tf.float32, shape=[None, 64, 64, 7], name="minimap")
        self.multi_select_in = tf.placeholder(tf.float32, shape=[None, 2, 7], name="multi_select")
        self.simple_action_in=tf.placeholder(tf.int32, shape=[None], name="simple_action_in")
        self.action_in=tf.placeholder(tf.int32, shape=[None, 7057], name="action_in")
        self.y_input = tf.placeholder("float", [None])
        self.session=tf.InteractiveSession()
        self.behaviour_net()
        self.target_net()
        self.loss()
        self.trainStep = tf.train.AdamOptimizer(self.lr).minimize(self.cost)
        self.session.run(tf.initialize_all_variables())
        self.update_target_net()
        tf.summary.scalar('cost', self.cost)
        self.summary_writer = tf.summary.FileWriter("C:/Users/probe/project_sc/add_guiyihua/logs", self.session.graph)
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("C:/Users/probe/project_sc/add_guiyihua/saved_networks")
        #print(checkpoint)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print ("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print ("Could not find old network weights")


    def step(self,timesteps):
        super(dqfd_less, self).step(timesteps)
        scr = timesteps.observation["screen"]
        minimap = timesteps.observation["minimap"]
        for i in range(17):
            scr[i]=scr[i]-np.ones((84,84))*scr[i].mean()
            scr[i]=scr[i]/(scr[i].max()+self.data.epsilon)
        for i in range(7):
            minimap[i] = minimap[i] - np.ones((64, 64)) * minimap[i].mean()
            minimap[i] = minimap[i] / (minimap[i].max() + self.data.epsilon)

        scr = scr.transpose(1,2,0)
        minimap = minimap.transpose(1,2,0)
        scr=scr.reshape(1,84,84,17)
        minimap=minimap.reshape(1,64,64,7)
        multi_select = np.zeros((2, 7))
        multi_select_raw = timesteps.observation["multi_select"]
        for i in range(min(multi_select_raw.shape[0], 2)):
            multi_select[i] = multi_select_raw[i]
        for i in range(7):
            multi_select[:,i]=multi_select[:,i]-np.ones((2))*multi_select[:,1].mean()
            multi_select[:,i]=multi_select[:,i]/(multi_select[:,i].max() + self.data.epsilon)

        multi_select = multi_select.reshape(1, 2, 7)
        player=timesteps.observation["player"]
        player=player.reshape(1,11)
        self.getaction(scr,minimap,multi_select)
        act=self.t
        if self.record ==2500:
            self.data.data_add_end(scr, minimap, multi_select, player, act)
        elif timesteps.step_type==0:
            self.data.data_add_first(scr,minimap,multi_select,player,act)
        else :
            self.data.data_add_middle(scr,minimap,multi_select,player,act)

        #if self.record%100 ==0:
        #    self.pretrain(100)


        return self.action_chosen




    def getaction(self,screen,minimap,multi_select):
        q=self.Q_behave.eval(
            feed_dict={self.screen_in: screen, self.minimap_in: minimap,
                       self.multi_select_in: multi_select})
        self.t=tf.argmax(q,1,output_type=tf.int32)[0]
        self.t=self.t.eval()
        if random.random()<self.greedy:
            if self.t!=7056 and multi_select[0][0][0]!=0:
                self.action_chosen= pysc2.lib.actions.FunctionCall(331,[[0],[self.t/84,self.t%84]])
            else:
                self.action_chosen=pysc2.lib.actions.FunctionCall(7,[[0]])
        else:
            self.t= random.randint(0,7056)
            if self.t!=7056 and multi_select[0][0][0]!=0:
                self.action_chosen= pysc2.lib.actions.FunctionCall(331,[[0],[self.t/84,self.t%84]])
            else:
                self.action_chosen=pysc2.lib.actions.FunctionCall(7,[[0]])
        self.record=self.record+1

    def target_net(self):
        init = tf.truncated_normal_initializer(stddev=0.1)
        zero_init = tf.zeros_initializer()
        screen_input = layers.InputLayer(self.screen_in, name="screen_inputs_t")
        screen_input_bn = layers.BatchNormLayer(screen_input, name="screen_bn_t")
        conv_depth = 64
        conv1_scr = layers.Conv2d(screen_input_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                  b_init=init,
                                  name="conv1_scr_t")
        conv1_scr_bn = layers.BatchNormLayer(conv1_scr, name="screen_bn_1_t")
        conv2_scr = layers.Conv2d(conv1_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                  b_init=init,
                                  name="conv2_scr_t")
        conv2_scr_bn = layers.BatchNormLayer(conv2_scr, name="screen_bn_2_t")
        conv3_scr = layers.Conv2d(conv2_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                  b_init=init,
                                  name="conv3_scr_t")
        conv3_scr_bn = layers.BatchNormLayer(conv3_scr, name="screen_bn_3_t")
        conv4_scr = layers.Conv2d(conv3_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                  b_init=init,
                                  name="conv4_scr_t")
        conv4_scr_bn = layers.BatchNormLayer(conv4_scr, name="screen_bn_4_t")
        conv5_scr = layers.Conv2d(conv4_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                  b_init=init,
                                  name="conv5_scr_t")
        conv5_scr_bn = layers.BatchNormLayer(conv5_scr, name="screen_bn_5_t")
        conv6_scr = layers.Conv2d(conv5_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                  b_init=init,
                                  name="conv6_scr_t")
        conv6_scr_bn = layers.BatchNormLayer(conv6_scr, name="screen_bn_6_t")
        conv7_scr = layers.Conv2d(conv6_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                  b_init=init,
                                  name="conv7_scr_t")
        conv7_scr_bn = layers.BatchNormLayer(conv7_scr, name="screen_bn_7_t")
        conv8_scr = layers.Conv2d(conv7_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                  b_init=init,
                                  name="conv8_scr_t")
        conv8_scr_bn = layers.BatchNormLayer(conv8_scr, name="screen_bn_8_t")
        conv9_scr = layers.Conv2d(conv8_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                  b_init=init,
                                  name="conv9_scr_t")
        conv9_scr_bn = layers.BatchNormLayer(conv9_scr, name="screen_bn_9_t")
        conv10_scr = layers.Conv2d(conv9_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init,
                                   name="conv10_scr_t")
        conv10_scr_bn = layers.BatchNormLayer(conv10_scr, name="screen_bn_10_t")
        scr_GAP = layers.PoolLayer(conv10_scr_bn, ksize=[1, 84, 84, 1], padding="VALID", pool=tf.nn.avg_pool,
                                   name="scr_GAP_t")
        scr_info = layers.FlattenLayer(scr_GAP, name="scr_flattened_t")

        minimap_input = layers.InputLayer(self.minimap_in, name="mini_in_t")
        minimap_input_bn = layers.BatchNormLayer(minimap_input, name="minimap_bn_t")
        conv1_mini = layers.Conv2d(minimap_input_bn, 32, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv1_mini_t")
        conv1_mini_bn = layers.BatchNormLayer(conv1_mini, name="mini_bn_1_t")
        conv2_mini = layers.Conv2d(conv1_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv2_mini_t")
        conv2_mini_bn = layers.BatchNormLayer(conv2_mini, name="mini_bn_2_t")
        conv3_mini = layers.Conv2d(conv2_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv3_mini_t")
        conv3_mini_bn = layers.BatchNormLayer(conv3_mini, name="mini_bn_3_t")
        conv4_mini = layers.Conv2d(conv3_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv4_mini_t")
        conv4_mini_bn = layers.BatchNormLayer(conv4_mini, name="mini_bn_4_t")
        conv5_mini = layers.Conv2d(conv4_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv5_mini_t")
        conv5_mini_bn = layers.BatchNormLayer(conv5_mini, name="mini_bn_5_t")
        conv6_mini = layers.Conv2d(conv5_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv6_mini_t")
        conv6_mini_bn = layers.BatchNormLayer(conv6_mini, name="mini_bn_6_t")
        conv7_mini = layers.Conv2d(conv6_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv7_mini_t")
        conv7_mini_bn = layers.BatchNormLayer(conv7_mini, name="mini_bn_7_t")
        conv8_mini = layers.Conv2d(conv7_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv8_mini_t")
        conv8_mini_bn = layers.BatchNormLayer(conv8_mini, name="mini_bn_8_t")
        conv9_mini = layers.Conv2d(conv8_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv9_mini_t")
        conv9_mini_bn = layers.BatchNormLayer(conv9_mini, name="mini_bn_9_t")
        conv10_mini = layers.Conv2d(conv9_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                    b_init=init, name="conv10_mini_t")
        conv10_mini_bn = layers.BatchNormLayer(conv10_mini, name="mini_bn_10_t")
        mini_GAP = layers.PoolLayer(conv10_mini_bn, ksize=[1, 64, 64, 1], padding="VALID", pool=tf.nn.avg_pool,
                                    name="mini_GAP_t")
        mini_info = layers.FlattenLayer(mini_GAP, name="mini_flattened_t")
        multi_select_in = layers.InputLayer(self.multi_select_in, name="multi_select_t")
        select_info = layers.FlattenLayer(multi_select_in, name="select_flattened_t")
        select_info_bn = layers.BatchNormLayer(select_info, name="select_bn_t")

        info_combined = layers.ConcatLayer([scr_info, mini_info, select_info_bn], name="info_all_t")
        dense1 = layers.DenseLayer(info_combined, n_units=1000, act=tf.nn.relu, W_init=init, b_init=init,
                                   name="Dense1_t")
        dense1_bn = layers.BatchNormLayer(dense1, name="dense1_bn_t")
        dense2 = layers.DenseLayer(dense1_bn, n_units=1000, act=tf.nn.relu, W_init=init, b_init=init,
                                   name="Dense2_t")
        dense2_bn = layers.BatchNormLayer(dense2, name="dense2_bn_t")
        dense3 = layers.DenseLayer(dense2_bn, n_units=1000, act=tf.nn.relu, W_init=init, b_init=init,
                                   name="Dense3_t")
        dense3_bn = layers.BatchNormLayer(dense3, name="dense3_bn_t")
        dense4 = layers.DenseLayer(dense3_bn, n_units=1000, act=tf.nn.relu, W_init=init, b_init=init,
                                   name="Dense4_t")
        dense4_bn = layers.BatchNormLayer(dense4, name="dense4_bn_t")
        dense5 = layers.DenseLayer(dense4_bn, n_units=1000, act=tf.nn.relu, W_init=init, b_init=init,
                                   name="Dense5_t")
        dense5_bn = layers.BatchNormLayer(dense5, name="dense5_bn_t")
        dense6 = layers.DenseLayer(dense5_bn, n_units=1000, act=tf.nn.relu, W_init=init, b_init=init,
                                   name="Dense6_t")
        dense6_bn = layers.BatchNormLayer(dense6, name="dense6_bn_t")
        self.q_target = layers.DenseLayer(dense6_bn, n_units=7057, W_init=zero_init, b_init=zero_init,
                                          name="q_t")
        self.target_params = self.q_target.all_params
        self.Q_target = self.q_target.outputs



    def behaviour_net(self):
        init = tf.truncated_normal_initializer(stddev=0.1)
        zero_init = tf.zeros_initializer()
        screen_input = layers.InputLayer(self.screen_in, name="screen_inputs")
        screen_input_bn = layers.BatchNormLayer(screen_input, name="screen_bn")
        conv_depth=64
        conv1_scr = layers.Conv2d(screen_input_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init, b_init=init,
                                  name="conv1_scr")
        conv1_scr_bn = layers.BatchNormLayer(conv1_scr, name="screen_bn_1")
        conv2_scr = layers.Conv2d(conv1_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init, b_init=init,
                                  name="conv2_scr")
        conv2_scr_bn = layers.BatchNormLayer(conv2_scr, name="screen_bn_2")
        conv3_scr = layers.Conv2d(conv2_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init, b_init=init,
                                  name="conv3_scr")
        conv3_scr_bn = layers.BatchNormLayer(conv3_scr, name="screen_bn_3")
        conv4_scr = layers.Conv2d(conv3_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init, b_init=init,
                                  name="conv4_scr")
        conv4_scr_bn = layers.BatchNormLayer(conv4_scr, name="screen_bn_4")
        conv5_scr = layers.Conv2d(conv4_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init, b_init=init,
                                  name="conv5_scr")
        conv5_scr_bn = layers.BatchNormLayer(conv5_scr, name="screen_bn_5")
        conv6_scr = layers.Conv2d(conv5_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init, b_init=init,
                                  name="conv6_scr")
        conv6_scr_bn = layers.BatchNormLayer(conv6_scr, name="screen_bn_6")
        conv7_scr = layers.Conv2d(conv6_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init, b_init=init,
                                  name="conv7_scr")
        conv7_scr_bn = layers.BatchNormLayer(conv7_scr, name="screen_bn_7")
        conv8_scr = layers.Conv2d(conv7_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init, b_init=init,
                                  name="conv8_scr")
        conv8_scr_bn = layers.BatchNormLayer(conv8_scr, name="screen_bn_8")
        conv9_scr = layers.Conv2d(conv8_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init, b_init=init,
                                  name="conv9_scr")
        conv9_scr_bn = layers.BatchNormLayer(conv9_scr, name="screen_bn_9")
        conv10_scr = layers.Conv2d(conv9_scr_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init, b_init=init,
                                  name="conv10_scr")
        conv10_scr_bn = layers.BatchNormLayer(conv10_scr, name="screen_bn_10")
        scr_GAP = layers.PoolLayer(conv10_scr_bn, ksize=[1, 84, 84, 1], padding="VALID", pool=tf.nn.avg_pool,
                                   name="scr_GAP")
        scr_info = layers.FlattenLayer(scr_GAP, name="scr_flattened")

        minimap_input = layers.InputLayer(self.minimap_in, name="mini_in")
        minimap_input_bn = layers.BatchNormLayer(minimap_input, name="minimap_bn")
        conv1_mini = layers.Conv2d(minimap_input_bn, 32, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv1_mini")
        conv1_mini_bn = layers.BatchNormLayer(conv1_mini, name="mini_bn_1")
        conv2_mini = layers.Conv2d(conv1_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv2_mini")
        conv2_mini_bn = layers.BatchNormLayer(conv2_mini, name="mini_bn_2")
        conv3_mini = layers.Conv2d(conv2_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv3_mini")
        conv3_mini_bn = layers.BatchNormLayer(conv3_mini, name="mini_bn_3")
        conv4_mini = layers.Conv2d(conv3_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv4_mini")
        conv4_mini_bn = layers.BatchNormLayer(conv4_mini, name="mini_bn_4")
        conv5_mini = layers.Conv2d(conv4_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv5_mini")
        conv5_mini_bn = layers.BatchNormLayer(conv5_mini, name="mini_bn_5")
        conv6_mini = layers.Conv2d(conv5_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv6_mini")
        conv6_mini_bn = layers.BatchNormLayer(conv6_mini, name="mini_bn_6")
        conv7_mini = layers.Conv2d(conv6_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv7_mini")
        conv7_mini_bn = layers.BatchNormLayer(conv7_mini, name="mini_bn_7")
        conv8_mini = layers.Conv2d(conv7_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv8_mini")
        conv8_mini_bn = layers.BatchNormLayer(conv8_mini, name="mini_bn_8")
        conv9_mini = layers.Conv2d(conv8_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv9_mini")
        conv9_mini_bn = layers.BatchNormLayer(conv9_mini, name="mini_bn_9")
        conv10_mini = layers.Conv2d(conv9_mini_bn, conv_depth, (3, 3), act=tf.nn.relu, padding="SAME", W_init=init,
                                   b_init=init, name="conv10_mini")
        conv10_mini_bn = layers.BatchNormLayer(conv10_mini, name="mini_bn_10")
        mini_GAP = layers.PoolLayer(conv10_mini_bn, ksize=[1, 64, 64, 1], padding="VALID", pool=tf.nn.avg_pool,
                                    name="mini_GAP")
        mini_info = layers.FlattenLayer(mini_GAP, name="mini_flattened")
        multi_select_in = layers.InputLayer(self.multi_select_in, name="multi_select")
        select_info = layers.FlattenLayer(multi_select_in, name="select_flattened")
        select_info_bn = layers.BatchNormLayer(select_info, name="select_bn")

        info_combined = layers.ConcatLayer([scr_info, mini_info, select_info_bn], name="info_all")
        dense1 = layers.DenseLayer(info_combined, n_units=1000, act=tf.nn.relu, W_init=init, b_init=init, name="Dense1")
        dense1_bn = layers.BatchNormLayer(dense1, name="dense1_bn")
        dense2 = layers.DenseLayer(dense1_bn, n_units=1000, act=tf.nn.relu, W_init=init, b_init=init,
                                   name="Dense2")
        dense2_bn = layers.BatchNormLayer(dense2, name="dense2_bn")
        dense3 = layers.DenseLayer(dense2_bn, n_units=1000, act=tf.nn.relu, W_init=init, b_init=init,
                                   name="Dense3")
        dense3_bn = layers.BatchNormLayer(dense3, name="dense3_bn")
        dense4 = layers.DenseLayer(dense3_bn, n_units=1000, act=tf.nn.relu, W_init=init, b_init=init,
                                   name="Dense4")
        dense4_bn = layers.BatchNormLayer(dense4, name="dense4_bn")
        dense5 = layers.DenseLayer(dense4_bn, n_units=1000, act=tf.nn.relu, W_init=init, b_init=init,
                                   name="Dense5")
        dense5_bn = layers.BatchNormLayer(dense5, name="dense5_bn")
        dense6 = layers.DenseLayer(dense5_bn, n_units=1000, act=tf.nn.relu, W_init=init, b_init=init,
                                   name="Dense6")
        dense6_bn = layers.BatchNormLayer(dense6, name="dense6_bn")
        self.q=layers.DenseLayer(dense6_bn, n_units=7057, W_init=zero_init, b_init=zero_init,
                                   name="q")
        self.Q_behave=self.q.outputs
    def test(self):
        print(self.q.all_params)

    def update_target_net(self):
        self.behave_params=self.q.all_params
        self.target_params=self.q_target.all_params
        return [tf.assign(e,s) for e,s in zip(self.target_params,self.behave_params)]

    def margin_loss(self):
        margin_loss=0
        loss_l = tf.one_hot(self.simple_action_in, depth=7057, on_value=0.0, off_value=10.0)
        jeq = tf.add(self.Q_behave, loss_l)
        action_best=tf.argmax(jeq,1,output_type=tf.int32)
        for i in range(self.batchsize):
            margin_loss=margin_loss+tf.square(jeq[i,action_best[i]]-self.Q_behave[i,self.simple_action_in[i]])/self.batchsize
        tf.summary.scalar("margin",margin_loss)
        self.cost=self.cost+margin_loss




    def loss(self):
        self.cost=0
        t=0
        for w in tl.layers.get_variables_with_name('conv1_scr') and tl.layers.get_variables_with_name('conv2_scr')and tl.layers.get_variables_with_name('conv3_scr') and tl.layers.get_variables_with_name('conv4_scr')and tl.layers.get_variables_with_name('conv5_scr')and tl.layers.get_variables_with_name('conv6_scr')and tl.layers.get_variables_with_name('conv7_scr')and tl.layers.get_variables_with_name('conv8_scr')and tl.layers.get_variables_with_name('conv9_scr')and tl.layers.get_variables_with_name('conv10_scr')and tl.layers.get_variables_with_name('conv1_mini')and tl.layers.get_variables_with_name('conv2_mini')and tl.layers.get_variables_with_name('conv3_mini')and tl.layers.get_variables_with_name('conv4_mini')and tl.layers.get_variables_with_name('conv5_mini')and tl.layers.get_variables_with_name('conv6_mini')and tl.layers.get_variables_with_name('conv7_mini')and tl.layers.get_variables_with_name('conv8_mini')and tl.layers.get_variables_with_name('conv9_mini')and tl.layers.get_variables_with_name('conv10_mini')and tl.layers.get_variables_with_name('Dense1')and tl.layers.get_variables_with_name('Dense2')and tl.layers.get_variables_with_name('Dense3')and tl.layers.get_variables_with_name('Dense4')and tl.layers.get_variables_with_name('Dense5')and tl.layers.get_variables_with_name('Dense6')and tl.layers.get_variables_with_name('q'):
            t=t+tf.contrib.layers.l2_regularizer(0.1)(w)
        self.cost=self.cost+t
        q_chosen=tf.reduce_sum(tf.multiply(self.Q_behave,tf.cast(self.action_in,tf.float32)))
        qloss=tf.reduce_mean(tf.square(tf.subtract(q_chosen,self.y_input)))
        self.cost=self.cost+qloss
        tf.summary.scalar("dqnloss",qloss)
        tf.summary.scalar("regularize",t)
        self.margin_loss()

    def pretrain(self,times):
        for m in range(times):
            permutation = np.random.permutation(self.data.data_pointer if self.data.data_pointer<self.data.next_data_pointer else self.data.next_data_pointer)
            screen_batch = self.data.screen[permutation, :, :, :][0:self.batchsize, :, :, :]
            minimap_batch = self.data.minimap[permutation, :, :, :][0:self.batchsize, :, :, :]
            screen_next_batch = self.data.screen_next[permutation, :, :, :][0:self.batchsize, :, :, :]
            minimap_next_batch = self.data.minimap_next[permutation, :, :, :][0:self.batchsize, :, :, :]
            multi_select_batch = self.data.multi_select[permutation, :, :][0:self.batchsize, 0:2, :]
            multi_select_next_batch = self.data.multi_select_next[permutation, :, :][0:self.batchsize, 0:2, :]
            actions_batch = self.data.action_actual[permutation][0:self.batchsize]
            score_batch = self.data.score[permutation][0:self.batchsize]
            simple_action_batch=self.data.action_simple[permutation][0:self.batchsize]

            y_batch = []
            QValue_batch = self.Q_target.eval(
                feed_dict={self.screen_in: screen_next_batch, self.minimap_in: minimap_next_batch,
                           self.multi_select_in: multi_select_next_batch})
            for i in range(0, self.batchsize):
                if i != self.batchsize - 1:
                    y_batch.append(score_batch[i] + self.gamma * np.max(QValue_batch[i]))
                else:
                    y_batch.append(score_batch[i])

            self.trainStep.run(feed_dict={self.screen_in: screen_batch,
                                          self.minimap_in: minimap_batch,
                                          self.multi_select_in: multi_select_batch,
                                          self.y_input: y_batch,
                                          self.action_in: actions_batch,
                                          self.simple_action_in:simple_action_batch})
            self.timestep=self.timestep+1
            if self.timestep %10 ==0:
                self.saver.save(self.session, 'C:/Users/probe/project_sc/add_guiyihua/saved_networks/' + 'network' + '-dqn', global_step=self.timestep)
                self.update_target_net()
                result = self.session.run(self.merged, feed_dict={self.screen_in: screen_batch,
                                          self.minimap_in: minimap_batch,
                                          self.multi_select_in: multi_select_batch,
                                          self.y_input: y_batch,
                                          self.action_in: actions_batch,
                                          self.simple_action_in:simple_action_batch})

                self.summary_writer.add_summary(result, self.timestep)
                print(self.timestep)

def main():
    net=dqfd_less()
    net.pretrain(10000)


if __name__=="__main__":
    main()






