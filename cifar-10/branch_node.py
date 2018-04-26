from __future__ import division
import tensorflow as tf
from utils import *

class branch_network():
	def __init__(self, image, target, num_class, name):
		self.image = image
		self.target = target
		self.num_class = num_class
		self.name = name
		self.keep_prob=tf.placeholder_with_default([1.0,1.0],shape=[2])
		self.train_mode = tf.placeholder_with_default(False, shape=())
		print(self.keep_prob)
		print(self.train_mode)

	def conv_bn(self, feature, filters, kernel):
		conv = tf.layers.conv2d(feature, filters, kernel, padding='same')
		bn = tf.layers.batch_normalization(conv, training=self.train_mode)
		return tf.nn.relu(bn)

	def net_structure(self):
		with tf.variable_scope(self.name):
			conv1 = self.conv_bn(self.image, 32, 5)
			pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)
			dropout1 = tf.layers.dropout(pool1, rate=self.keep_prob[0])
			conv2 = self.conv_bn(dropout1, 64, 5)
			pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)
			dropout2 = tf.layers.dropout(pool2,rate=self.keep_prob[0])

			conv3 = self.conv_bn(dropout2, 64, 3)
			pool3 = tf.layers.average_pooling2d(conv3, pool_size=2, strides=2)

			pool3_dropout = tf.layers.dropout(pool3, rate=self.keep_prob[0])
			fc1 = tf.layers.flatten(pool3_dropout)
			# print(fc1)
			fc2 = tf.layers.dense(fc1, 128, activation=tf.nn.relu)
			fc2_dropout = tf.layers.dropout(fc2, rate=self.keep_prob[1])
			print(fc2_dropout)
		fc3 = tf.layers.dense(fc2_dropout, self.num_class)
		return fc3

	def train(self):
		logits = self.net_structure()

		cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=self.target, logits=logits))
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			optimizer = tf.train.AdamOptimizer().minimize(cross_entropy)
		correct = tf.equal(tf.argmax(tf.nn.softmax(logits),1), self.target)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.name)
		g_list = tf.global_variables()
		bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
		bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
		var_list += bn_moving_vars
		# saver = tf.train.Saver(var_list)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			dataset=data_process()
			for epoch in range(30):
				x_batch,y_batch=dataset.fine_tune_next_batch(128,[3,5,7,2,4,6])
				for step in range(len(x_batch)):
					loss,_,acc=sess.run([cross_entropy,optimizer,accuracy],
										feed_dict={self.image:x_batch[step],self.target:y_batch[step],
												   self.keep_prob:[0.25,0.5],self.train_mode:True})
					if step%10==9:
						# saver.save(sess,self.name+'_initial_variables/branch.module',global_step=epoch*len(x_batch)+step)
						print("number epoch %d,number step %d,cross entropy is %f"%(epoch,step,loss))
						print("number epoch %d,number step %d,accuracy is %f"%(epoch,step,acc))
				test_accuracy=0
				x_batch,y_batch=dataset.fine_tune_next_batch(128,[3,5,7,2,4,6],mode='test')
				for step in range(len(x_batch)):
					test_acc=sess.run(accuracy,feed_dict={self.image:x_batch[step],self.target:y_batch[step]})
					test_accuracy+=test_acc
				print("test accuray is %f"%(test_accuracy/len(x_batch)))


# image=tf.placeholder(tf.float32,[None,32,32,3])
# target=tf.placeholder(tf.int64,[None])
# # num_class=6
# name='branch'
# net=branch_network(image,target,6,name)
# net.train()
# net.net_structure()
# print(net)


