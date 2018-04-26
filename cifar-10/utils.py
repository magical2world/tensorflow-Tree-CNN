import os
import cPickle
from matplotlib.font_manager import *
import numpy as np

class data_process():
	def __init__(self):
		print("Load data,please wait for seconds")
		folder_path='/home/liushaolin/Tree-CNN/cifar-10-batches-py'
		self.train_images,self.train_labels,self.test_images,self.test_labels=self.load_data(folder_path)
		print("Load data successful")
	def unpickle(self,file):
		with open(file,'rb') as fo:
			dict=cPickle.load(fo)
		return dict
	def load_data(self,path):
		train_images=np.empty((0,32,32,3))
		train_labels=np.empty((0,1))
		test_images=np.empty((0,32,32,3))
		test_labels=np.empty((0,1))
		for filename in os.listdir(path):
			file=path+'/'+filename
			try:
				image=self.unpickle(file)['data']
				label=np.array(self.unpickle(file)['labels']).reshape([-1,1])
				image=np.transpose(image.reshape([-1, 3, 32, 32]),[0, 2, 3, 1])
				image=image/255.0
				if filename[:4]=='data':
					train_images=np.vstack((train_images,image))
					train_labels=np.vstack((train_labels,label))
				if filename[:4]=='test':
					test_images=np.vstack((test_images,image))
					test_labels=np.vstack((test_labels,label))
			except:
				continue
		return train_images,train_labels,test_images,test_labels

	def next_batch(self,batch_size,node='root',mode='train'):
		x_data=np.empty((0,32,32,3))
		y_data=np.empty((0))
		if node=='root':
			node1_class=[3,5,7]
			node2_class=[1,8,9]
			if mode=='train':
				for num in node1_class:
					x_data=np.vstack((x_data,self.train_images[np.where(self.train_labels==num)[0]]))
				y_data=np.hstack((y_data,np.zeros(len(x_data))))

				for num in node2_class:
					x_data=np.vstack((x_data,self.train_images[np.where(self.train_labels==num)[0]]))
				y_data=np.hstack((y_data,np.zeros(len(x_data))+1))

			else:
				for num in node1_class:
					x_data=np.vstack((x_data,self.test_images[np.where(self.test_labels==num)[0]]))
				class_length=len(x_data)
				y_data=np.hstack((y_data,np.zeros(class_length)))
				for num in node2_class:
					x_data=np.vstack((x_data,self.test_images[np.where(self.test_labels==num)[0]]))
				y_data=np.hstack((y_data,np.zeros(class_length)+1))

		elif node=='branch1':
			if mode=='train':
				for idx,num in enumerate([3,5,7]):
					x_data=np.vstack((x_data,self.train_images[np.where(self.train_labels==num)[0]]))
					y_data=np.vstack((y_data,np.zeros(len(self.train_images[np.where(self.train_labels==num)[0]]))+idx))
			else:
				for idx,num in enumerate([3,5,7]):
					x_data=np.vstack((x_data,self.test_images[np.where(self.test_labels==num)[0]]))
					y_data=np.vstack((y_data,np.zeros(len(self.test_images[np.where(self.test_labels==num)[0]]))+idx))

		elif node=='branch2':
			if mode=='train':
				for idx,num in enumerate([3,5,7]):
					x_data=np.vstack((x_data,self.train_images[np.where(self.train_labels==num)[0]]))
					y_data=np.vstack((y_data,np.zeros(len(self.train_images[np.where(self.train_labels==num)[0]]))+idx))
			else:
				for idx, num in enumerate([1,8,9]):
					x_data=np.vstack((x_data,self.test_images[np.where(self.test_labels==num)[0]]))
					y_data=np.vstack((y_data,np.zeros(len(self.test_images[np.where(self.test_labels==num)[0]]))+idx))

		size=len(x_data)
		index=np.arange(0,size)
		np.random.shuffle(index)
		x_in=x_data[index]
		y_in=y_data[index]
		x_batch=[]
		y_batch=[]
		start_index=0
		while(1):
			end_index=start_index+batch_size
			if end_index>size:
				break
			x_batch.append(x_in[start_index:end_index])
			y_batch.append(y_in[start_index:end_index])
			start_index=end_index
		return x_batch,y_batch

	def fine_tune_next_batch(self,batch_size,branch_class,mode='train'):
		image=np.empty((0, 32, 32, 3))
		label=np.empty((0))
		if mode=='train':
			for idx,class_num in enumerate(branch_class):
				num_image=self.train_images[np.where(self.train_labels==class_num)[0]]
				image=np.vstack((image,num_image))
				label=np.hstack((label,np.zeros(len(num_image))+idx))
		else:
			for idx,class_num in enumerate(branch_class):
				num_image=self.test_images[np.where(self.test_labels==class_num)[0]]
				image=np.vstack((image,num_image))
				label=np.hstack((label,np.zeros(len(num_image))+idx))
		length=len(image)
		idx=np.arange(0,length)
		np.random.shuffle(idx)
		image=image[idx]
		label=label[idx]
		x_batch=[]
		y_batch=[]
		start_index=0
		while(1):
			end_index=start_index+batch_size
			if end_index>length:
				break
			x_batch.append(image[start_index:end_index])
			y_batch.append(label[start_index:end_index])
			start_index=end_index
		return x_batch,y_batch

# dataset=data_process()
# x_batch,y_batch=dataset.next_batch(128,mode='test')
# print(len(x_batch[1]))
# import matplotlib.pyplot as plt
#
# plt.subplot(221)
# plt.imshow(x_batch[30][1])
# print(y_batch[30][1])
# plt.subplot(222)
# plt.imshow(x_batch[30][2])
# print(y_batch[30][2])
# plt.subplot(223)
# plt.imshow(x_batch[30][3])
# print(y_batch[30][3])
# plt.subplot(224)
# plt.imshow(x_batch[30][4])
# print(y_batch[30][4])
# plt.show()
