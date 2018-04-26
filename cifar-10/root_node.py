from __future__ import division
import tensorflow as tf
from utils import *
class root_network():
	def __init__(self,image,target,root_class=2):
		self.image=image
		self.target=target
		self.root_class=root_class
		self.keep_prob=tf.placeholder_with_default(1,shape=())
		self.train_mode=tf.placeholder_with_default(False,shape=())
	def conv_bn(self,feature,filters,kernel):
		conv=tf.layers.conv2d(feature,filters,kernel,padding='same')
		bn=tf.layers.batch_normalization(conv,training=self.train_mode)
		return tf.nn.relu(bn)
	def net_structure(self):
		conv1=self.conv_bn(self.image,64,5)
		pool1=tf.layers.max_pooling2d(conv1,pool_size=2,strides=2)
		conv2_1=self.conv_bn(pool1,128,3)
		dropout1=tf.layers.dropout(conv2_1,self.keep_prob)
		conv2_2=self.conv_bn(dropout1,128,3)
		pool2=tf.layers.max_pooling2d(conv2_2,pool_size=2,strides=2)

		fc1=tf.layers.flatten(pool2)
		fc2=tf.layers.dense(fc1,512,activation=tf.nn.relu)
		fc2_dropout=tf.layers.dropout(fc2,self.keep_prob)
		fc3=tf.layers.dense(fc2_dropout,128,activation=tf.nn.relu)
		fc3_dropout=tf.layers.dropout(fc3,self.keep_prob)
		fc4=tf.layers.dense(fc3_dropout,self.root_class)
		# print(fc4)
		return fc4

	def train(self):
		logits=self.net_structure()
		cross_entropy=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=self.target,logits=logits))
		update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			optimizer=tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
		correct=tf.equal(tf.argmax(logits,1),self.target)
		accuracy=tf.reduce_mean(tf.cast(correct,tf.float32))
		var_list=tf.trainable_variables()
		g_list=tf.global_variables()
		bn_moving_vars=[g for g in g_list if 'moving_mean' in g.name]
		bn_moving_vars+=[g for g in g_list if 'moving_variance' in g.name]
		var_list+=bn_moving_vars
		saver=tf.train.Saver(var_list)
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			# saver=tf.train.Saver()
			dataset=data_process()
			for epoch in range(5):
				x_batch,y_batch=dataset.fine_tune_next_batch(128)
				for step in range(len(x_batch)):
					loss,_,acc=sess.run([cross_entropy,optimizer,accuracy],
										feed_dict={self.image:x_batch[step],self.target:y_batch[step],
												   self.keep_prob:0.5,self.train_mode:True})
					if step%10==9:
						# saver.save(sess,'root_initial_variables/root.module',global_step=epoch*len(x_batch)+step)
						print("number epoch %d,number step %d,cross entropy is %f"%(epoch,step,loss))
						print("number epoch %d,number step %d,accuracy is %f"%(epoch,step,acc))
				x_batch,y_batch=dataset.next_batch(100,mode='test')
				test_accuracy=0
				for step in range(len(x_batch)):
					test_acc=sess.run(accuracy,feed_dict={self.image:x_batch[step],self.target:y_batch[step]})
					test_accuracy+=test_acc
				print('test accuracy is %f'%(test_accuracy/len(x_batch)))

	# def restore(self,new_train_data):
	# 	logits=self.net_structure()
	# 	with tf.Session() as sess:
	# 		sess.run(tf.global_variables_initializer())
	# 		saver=tf.train.Saver()
	#

# image=tf.placeholder(tf.float32,[None,32,32,3])
# target=tf.placeholder(tf.int64,[None])
#
# net=root_network(image,target)
# net.net_structure()