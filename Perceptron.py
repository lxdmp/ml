# coding=utf-8

import copy

class Perceptron(object):
	'''
	实现一感知器,需指定数据的维度与激活函数.
	'''
	def __init__(self, input_dimension, activator):
		self.w0 = 0.0 # 以两部分表示权重,如此无需拼接数据
		self.weights = [0.0 for i in range(input_dimension)]
		self.activator = activator

	def __str__(self):
		w = [self.w0]
		w.extend(self.weights)
		return str(w)

	def predict(self, data):
		if len(self.weights)!=len(data):
			raise Exception('weight dimension(%d) not matched with data dimension(%d)', \
				len(self.weights), len(data))

		val = reduce(lambda x,y : x+y, \
				map(lambda(w,x) : w*x, zip(self.weights, data)))+self.w0
		return self.activator(val)
	
	def train(self, data_set, flag_set, train_times, train_rate, trace=False):
		w0_backup = self.w0
		weight_backup = copy.copy(self.weights)
		for i in range(train_times):
			self.train_once(data_set, flag_set, train_rate)
			if cmp(weight_backup, self.weights)==0 and w0_backup==self.w0:
				break
			w0_backup = self.w0
			weight_backup = copy.copy(self.weights)
			if trace:
				print str(self)

	def train_once(self, data_set, flag_set, train_rate):
		'''
		以'批处理梯度下降方法'进行一次训练
		'''
		for data,flag in zip(data_set, flag_set):
			delta = flag-self.predict(data)
			self.weights = map(lambda(w,d) : w+train_rate*delta*d, zip(self.weights, data))
			self.w0 += train_rate*delta

if __name__=='__main__':
	data = [[0,0], [0,1], [1,0], [1,1]]
	flag = [0, 0, 0, 1]
	m = Perceptron(2, lambda x : 1 if x>0 else 0)
	m.train(data, flag, 10, 0.1)
	print m
	for item in data:
		print '%d and %d : %d' % (item[0], item[1], m.predict(item))

	data = [[5], [3], [8], [1.4], [10.1]]
	flag = [5500, 2300, 7600, 1800, 11400]
	m = Perceptron(1, lambda x:x)
	m.train(data, flag, 10, 0.01)
	print m
	test_data = [[3.4], [15], [1.5], [6.3]]
	for test in test_data:
		print '%.2f -> %.2f' % (test[0], m.predict(test))

