# TSE YOUR CODE FOR THE SECOND TASK OF TODO.TXT!

import numpy as np
import tensorflow as tf

a0=tf.placeholder(tf.float32,[None, 1])
a1=tf.placeholder(tf.float32,[None, 1])
label=tf.placeholder(tf.float32,[None, 1])

a2=tf.zeros_like(a0)
#YOUR CODE HERE
#YOU SHOULD REPLACE TENSOR a2 :
#  a2[i,0] = a0[i,0] if label[i,0] == 0 else a0[i,0] +a1[i,0]
#SO THE RESULT SHOULD BE: 0.0 3.0 6.0 3.0









#########################################

input_0=np.array([[0.0], [1.0],[2.0],[3.0]])
input_1=np.array([[1.0], [2.0],[4.0],[3.0]])
input_label=np.array([[0.0], [1.0],[1.0],[0.0]])
with tf.Session() as sess:
    outcome = sess.run([a2],feed_dict={a0:input_0,a1:input_1,label:input_label})
    print(outcome)

