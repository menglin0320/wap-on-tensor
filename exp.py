import numpy as np
import tensorflow as tf

fake_in = tf.zeros((3,233),dtype = tf.float32)
fake_in_2 = tf.zeros((3,233),dtype = tf.float32)
lstm = tf.contrib.rnn.BasicLSTMCell(233) 
state = lstm.zero_state(3,dtype = tf.float32)
with tf.variable_scope("decoder",reuse=None):
	out, state = lstm(fake_in, state)
with tf.variable_scope("decoder", reuse=True):
	out2, state = lstm(fake_in_2,state)

init = tf.global_variables_initializer()
sess = tf.InteractiveSession(config=config)
[res1,res2]=sess.run(out,out2, feed_dict={self.x: X})
print(state)