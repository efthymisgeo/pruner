import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class RandomDataset(Dataset):
	def __init__(self, data_tuple):

		self.X, self.y = data_tuple

	def __len__(self):
		return(len(self.y))

	def __getitem__(self, index):
		return (self.X[index, :], self.y[index])


class CreateRandomDataset():
	""" create random dataset 
	"""
	def __init__(self, datatype="random", feat_size=300,
				 n_samples=1000, n_classes=3, val_ratio=0.2,
				 test_ratio=0.2, batch_size=32, labels_per_sample=1):
		self.type = datatype
		self.feat_size = feat_size
		self.samples = n_samples
		self.val_ratio = val_ratio
		self.test_ratio = test_ratio
		self.bs = batch_size
		self.labels = labels_per_sample

		if isinstance(n_classes, list):
			self.classes = n_classes
		else:
			self.classes = [n_classes 
							for _ in range(labels_per_sample)]

		self.data_dict = self.generate_dataset()
		
	def get_dataloaders(self):
		train_set = self.data_dict["train"]
		val_set = self.data_dict["val"]
		test_set = self.data_dict["test"]

		train_loader = DataLoader(RandomDataset(train_set),
								  batch_size=self.bs,
								  shuffle=True)
		val_loader = DataLoader(RandomDataset(val_set),
								batch_size=self.bs,
								shuffle=False)
		test_loader = DataLoader(RandomDataset(test_set),
								 batch_size=self.bs,
								 shuffle=False)

		return train_loader, val_loader, test_loader 

	def generate_dataset(self):
		if self.type == 'random':
			X, y = self._create_random_dataset()
		elif self.type == 'pseudo':
			X, y = self._create_pseudo_dataset()
		elif self.type == 'multilabel':
			X, y = self._create_multi_dataset()
		elif self.type == 'inv hierlabel':
			X, y = self._create_inv_hier_multi_dataset()
		elif self.type == 'hierlabel':
			X, y = self._create_hier_multi_dataset()
		elif self.type == 'sum_hierlabel':
			X, y = self._create_sum_multi_dataset()
		else:
			raise ValueError('Not an implemented dataset')
		
		X_temp, X_test, y_temp, y_test = \
			train_test_split(X, y, test_size=self.test_ratio, random_state=1)
		X_train, X_val, y_train, y_val = \
			train_test_split(X_temp, y_temp, test_size=self.val_ratio, random_state=1)

		return ({"train":(X_train,y_train),"val": (X_val,y_val),"test": (X_test,y_test)})


	def _create_random_dataset(self):
		X = np.random.rand(self.samples, self.feat_size)
		y = []
		# todo
		# fix for self.class is list case
		for _ in range(self.samples):
			y.append(np.random.randint(0,self.classes))
		return X, y

	def _create_pseudo_dataset(self):
		X = np.random.rand(self.samples, self.feat_size)
		y = []
		# TODO 
		# fix for self.class is list case
		for i in range(self.samples):
			label = np.random.randint(0,self.classes)
			y.append(label)
			# 3 is the label's position in the feature vector
			X[i,3] = label
		return X, y
	
	def _create_multi_dataset(self):
		X = np.random.rand(self.samples, self.feat_size)
		y = []
		for i in range(self.samples):
			label = []
			for l in range(self.labels):
				pseudo_label = np.random.randint(0, self.classes[l])
				label.append(pseudo_label)
				X[i,l] = pseudo_label
			y.append(label)	

		return X, y

	def _create_inv_hier_multi_dataset(self):
		X = np.random.rand(self.samples, self.feat_size)
		y = []
		for i in range(self.samples):
			label = []
			for l in range(self.labels):
				pseudo_label = np.random.randint(0, self.classes[l])
				label.append(pseudo_label)

			new_label = []
			for l in range(self.labels):
				if l == 0:
					new_label.append(sum(label[1:]))
				else:
					new_label.append(label[l]) 
				X[i,l] = new_label[l]
			y.append(new_label)	

		return X, y

	def _create_hier_multi_dataset(self):
		X = np.random.rand(self.samples, self.feat_size)
		y = []
		for i in range(self.samples):
			label = []
			for l in range(self.labels-1):
				pseudo_label = np.random.randint(0, self.classes[l])
				label.append(pseudo_label)
				X[i,l] = pseudo_label
			# last label is the sum of all previous
			pseudo_label = sum(label)
			label.append(pseudo_label)
			X[i, self.labels-1] = pseudo_label
			y.append(label) 

		return X, y

	def _create_sum_multi_dataset(self):
		X = np.random.rand(self.samples, self.feat_size)
		y = []
		for i in range(self.samples):
			label = []
			sum = 0
			for l in range(self.labels):
				pseudo_label = np.random.randint(0, self.labels)
				sum += pseudo_label
				label.append(sum)
				X[i,l] = pseudo_label
			y.append(label) 

		return X, y

	def dataset_statistics(self):
		#TODO
		pass

