import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnns
import tensorflow.contrib.layers as layers

def layer_stack(in_map, mask, n_layers,n_channels,last = False,is_training = True):
	convs = [in_map]
	for i in range(0,n_layers):
		conv = tf.multiply(tf.nn.relu(layers.batch_norm(layers.conv2d(convs[-1], num_outputs=n_channels, kernel_size=3,activation_fn = None,
                        stride=1, padding = 'SAME'),is_training = is_training,decay = 0.9)),mask)
		if last:
			conv = layers.dropout(conv,keep_prob = 0.8,is_training = is_training)
		convs.append(conv)

   	return convs[-1] 

class MathFormulaRecognizer():
	def __init__ (self,num_label,dim_hidden,device):
		self.num_label = num_label	
		self.dim_hidden = dim_hidden
		self.coverage_depth = 128
		self.dim_embed = 128
		self.is_train = tf.placeholder(tf.bool)
		self.x = tf.placeholder(tf.float32, [None,None,None,1]) 
		self.batch_size = tf.shape(self.x)[0]
		self.in_height = tf.shape(self.x)[1]
		self.in_width = tf.shape(self.x)[2]
		self.x_mask = tf.placeholder(tf.float32,[None,None,None])
		self.ex_mask = tf.expand_dims(self.x_mask,3)
		self.y = tf.placeholder(tf.int32, [None,None])
		self.y_mask = tf.placeholder(tf.float32,[None,None])
		self.seq_length = tf.shape(self.y)[1]
		#encoder:
		with tf.variable_scope('Encoder'):
			first_stack = layer_stack(self.x, self.ex_mask, 3,32,is_training= self.is_train)
			pooled = layers.max_pool2d(first_stack,2,2,padding = 'VALID')
			self.sx_mask = layers.max_pool2d(self.ex_mask,2,2,padding = 'VALID')
			# pooled = tf.multiply(pooled,self.sx_mask)

			second_stack = layer_stack(pooled, self.sx_mask, 3,64,is_training= self.is_train)
			pooled = layers.max_pool2d(second_stack,2,2,padding = 'VALID')
			self.sx_mask = layers.max_pool2d(self.sx_mask,2,2,padding = 'VALID')
			# pooled = tf.multiply(pooled,self.sx_mask)

			third_stack = layer_stack(pooled, self.sx_mask, 3,64,is_training= self.is_train)
			pooled = layers.max_pool2d(third_stack,2,2,padding = 'VALID')
			self.sx_mask = layers.max_pool2d(self.sx_mask,2,2,padding = 'VALID')
			# pooled = tf.multiply(pooled,self.sx_mask)

			fourth_stack = layer_stack(pooled, self.sx_mask, 3,128,True,self.is_train)
			pooled = layers.max_pool2d(fourth_stack,2,2,padding = 'VALID')
			self.sx_mask = layers.max_pool2d(self.sx_mask,2,2,padding = 'VALID')
			# pooled = tf.multiply(pooled,self.sx_mask)
			self.information_tensor = pooled

			
		#Decoder:
		self.latent_depth = 128
		self.attention_dimension = self.latent_depth
		self.feature_height = tf.shape(self.information_tensor)[1]
		self.feature_width = tf.shape(self.information_tensor)[2]
		self.feature_size =  self.feature_height* self.feature_width
		self.mean_feature = tf.reduce_mean(self.information_tensor ,axis = [1,2])
		self.vec_mask = tf.reshape(self.sx_mask,[-1,self.feature_size])	
			
		with tf.variable_scope('Decoder'):
			self.gru = rnns.GRUCell(self.dim_hidden)
			self.w_hidden = tf.get_variable("w_hidden", shape=[self.dim_hidden, self.attention_dimension],
								initializer=tf.contrib.layers.xavier_initializer())

			self.bias_hidden = tf.get_variable("bias_hidden", shape=[self.attention_dimension],
								initializer=tf.zeros_initializer())

			self.w_annotation =tf.get_variable("w_annotation", shape=[self.latent_depth, self.attention_dimension],
								initializer=tf.contrib.layers.xavier_initializer())
			self.bias_annotation = tf.get_variable("bias_annotation", shape=[self.attention_dimension],
								initializer=tf.zeros_initializer())

			self.w_f = tf.get_variable("w_f", shape=[self.coverage_depth, self.attention_dimension],
								initializer=tf.contrib.layers.xavier_initializer())

			self.bias_f = tf.get_variable("bias_f", shape=[self.attention_dimension],
								initializer=tf.zeros_initializer())

			self.w_2e = tf.get_variable("w_2e", shape=[self.attention_dimension, 1],
								initializer=tf.contrib.layers.xavier_initializer())
			self.bias_2e = tf.get_variable("bias_2e", shape=[1],
								initializer=tf.zeros_initializer())

			# self.w_init2c = tf.get_variable("w_init2c", shape=[self.latent_depth,self.latent_depth],
			# 					initializer=tf.contrib.layers.xavier_initializer())
			# self.bias_init2c = tf.get_variable("bias_init2c", shape=[self.latent_depth],
			# 					initializer=tf.zeros_initializer())
			
			self.w_init2hid = tf.get_variable('w_init2hid', shape=[self.latent_depth,self.dim_hidden],
								initializer=tf.contrib.layers.xavier_initializer())
			self.bias_init2hid = tf.get_variable("bias_init2hid", shape=[self.dim_hidden],
								initializer=tf.zeros_initializer())

			self.w_B2f_filter = tf.get_variable("w_B2f_filter", shape=[11, 11,1,self.coverage_depth],initializer=layers.xavier_initializer())

			self.w_2logit = tf.get_variable('w_2logit', shape=[self.dim_hidden,self.num_label],
								initializer=tf.contrib.layers.xavier_initializer())
			self.bias_2logit =  tf.get_variable("bias_2logit", shape=[self.num_label],
								initializer=tf.zeros_initializer())

			self.w_embedding = tf.get_variable('w_embedding', shape=[self.num_label, self.dim_embed])
			self.bias_embedding = tf.get_variable("bias_embedding", shape=[self.dim_embed],
								initializer=tf.zeros_initializer())

			self.counter_dis = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

	def build_train(self):
		#initialization:
		alpha_t = tf.ones([self.batch_size,self.feature_size],dtype = tf.float32)/ tf.cast(self.feature_size,tf.float32)
		beta_t = tf.zeros([self.batch_size,self.feature_size],dtype = tf.float32)
		# c = tf.matmul(self.mean_feature,self.w_init2c) + self.bias_init2c 
		state = tf.matmul(self.mean_feature,self.w_init2hid) + self.bias_init2hid
		
		total_loss = tf.constant(0.0,dtype = tf.float32)
		with tf.variable_scope('Decoder'):
			
			beta_t = beta_t + alpha_t
			F = tf.nn.conv2d(tf.reshape(beta_t,[-1,self.feature_height,self.feature_width,1]),self.w_B2f_filter,strides = [1,1,1,1],padding = 'SAME')
			F = tf.multiply(F,self.sx_mask)
			out = state
			#weighted x has shape [batchsize* feature size, attention dimension]
			weighted_h = tf.matmul(out, self.w_hidden) + self.bias_hidden
			weighted_h = tf.tile(weighted_h,[1,self.feature_size])
			weighted_h = tf.reshape(weighted_h,[-1,self.attention_dimension])
			# print('weighted_h',weighted_h.get_shape())
 			weighted_annotation =  tf.matmul(tf.reshape(self.information_tensor,[-1,self.latent_depth]),self.w_annotation) + self.bias_annotation

 			# print('weighted_annotation',weighted_annotation.get_shape())
			weighted_f = tf.matmul(tf.reshape(F,[-1,self.coverage_depth]),self.w_f) + self.bias_f
			# print('weighted_f',weighted_f.get_shape())
			SUM = tf.add(tf.add(weighted_h,weighted_annotation),weighted_f)
			# SUM = tf.reshape(SUM,[-1,self.feature_size,self.attention_dimension])
			# SUM = tf.transpose(SUM,[0,2,1])
			# SUM = tf.multiply(SUM,self.vec_mask)

			e = tf.matmul(tf.nn.tanh(SUM), self.w_2e) 
			e = tf.reshape(e,[-1,self.feature_size])+ self.bias_2e

			# print('e',e.get_shape())
			alpha_t = tf.exp(e)
			alpha_t = tf.multiply(alpha_t,self.vec_mask)
			alpha_t = alpha_t/tf.expand_dims(tf.reduce_sum(alpha_t,axis = -1),1)
			# alpha_t = tf.nn.softmax(e)
			# print('alpha_t',alpha_t.get_shape())
			c = tf.reduce_sum(tf.multiply(tf.transpose(tf.reshape(self.information_tensor,[-1,self.feature_size,self.latent_depth]),[2,0,1]),alpha_t),axis = -1)
			c = tf.transpose(c,[1,0])
			# expanded_alpha_t  = tf.tile(tf.expand_dims(alpha_t,2),[1,1,self.latent_depth])
			# print('expanded_alpha_t',expanded_alpha_t.get_shape())
			word_embedding = tf.nn.embedding_lookup(self.w_embedding,tf.tile(tf.constant([111,]),[self.batch_size])) + self.bias_embedding
			# print('word_embedding',word_embedding.get_shape())
			gru_in = tf.concat([c,word_embedding],axis = 1)
			out, state = self.gru(gru_in,state)

			labels = self.y[:,0]
			labels = tf.expand_dims(labels,1)
			batch_range = tf.expand_dims(tf.range(0,self.batch_size,1),1)
			sparse = tf.concat([batch_range,labels],1)
			onehot = tf.sparse_to_dense(sparse, tf.stack([self.batch_size, self.num_label]), 1.0, 0.0)
			logit = tf.matmul(out,self.w_2logit) + self.bias_2logit
			self.logit = logit
			xentropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit, labels = onehot)
			self.xentropy = xentropy * self.y_mask[:,0]

			loss = tf.reduce_sum(self.xentropy)
			total_loss += loss
			#first_round
			i = tf.constant(1)
			while_condition = lambda N1,i,N2,N3,N4,N5: tf.less(i, self.seq_length)
			tf.get_variable_scope().reuse_variables()	

			def body(total_loss,i,beta_t,state,alpha_t,out):
				beta_t = beta_t + alpha_t
				F = tf.nn.conv2d(tf.reshape(beta_t,[-1,self.feature_height,self.feature_width,1]),self.w_B2f_filter,strides = [1,1,1,1],padding = 'SAME')
				
				#weighted x has shape [batchsize* feature size, attention dimension]
				weighted_h = tf.matmul(out, self.w_hidden) + self.bias_hidden
				weighted_h = tf.tile(weighted_h,[1,self.feature_size])
				weighted_h = tf.reshape(weighted_h,[-1,self.attention_dimension])
				# print('weighted_h',weighted_h.get_shape())
	 			weighted_annotation =  tf.matmul(tf.reshape(self.information_tensor,[-1,self.latent_depth]),self.w_annotation) + self.bias_annotation
	 			# print('weighted_annotation',weighted_annotation.get_shape())
				weighted_f =   tf.matmul(tf.reshape(F,[-1,self.coverage_depth]),self.w_f) + self.bias_f
				# print('weighted_f',weighted_f.get_shape())
				e = tf.matmul(tf.nn.tanh(tf.add(tf.add(weighted_h,weighted_annotation),weighted_f)), self.w_2e) 
				e = tf.reshape(e,[-1,self.feature_size])+ self.bias_2e
				# print('e',e.get_shape())
				# alpha_t = tf.nn.softmax(e)
				alpha_t = tf.exp(e)
				alpha_t = tf.multiply(alpha_t,self.vec_mask)
				alpha_t = alpha_t/tf.expand_dims(tf.reduce_sum(alpha_t,axis = -1),1)
				# print('alpha_t',alpha_t.get_shape())

				# expanded_alpha_t  = tf.tile(tf.expand_dims(alpha_t,2),[1,1,self.latent_depth])
				# print('expanded_alpha_t',expanded_alpha_t.get_shape())
				c = tf.reduce_sum(tf.multiply(tf.transpose(tf.reshape(self.information_tensor,[-1,self.feature_size,self.latent_depth]),[2,0,1]),alpha_t),axis = -1)
				c = tf.transpose(c,[1,0])
				word_embedding = tf.nn.embedding_lookup(self.w_embedding,self.y[:,i-1]) + self.bias_embedding
				# print('c',c.get_shape())
				# print('word_embedding',word_embedding.get_shape())
				gru_in = tf.concat([c,word_embedding],axis = 1)
				out, state = self.gru(gru_in,state)

				labels = self.y[:,i]
				labels = tf.expand_dims(labels,1)
				batch_range = tf.expand_dims(tf.range(0,self.batch_size,1),1)
				sparse = tf.concat([batch_range,labels],1)
				onehot = tf.sparse_to_dense(sparse, tf.stack([self.batch_size, self.num_label]), 1.0, 0.0)
				logit = tf.matmul(out,self.w_2logit) + self.bias_2logit
				xentropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit, labels = onehot)
				xentropy = xentropy * self.y_mask[:,i]
				loss = tf.reduce_sum(xentropy)
				total_loss += loss
				return [total_loss,tf.add(i, 1),beta_t,state,alpha_t,out]

			# do the loop:
			[total_loss,i,beta_t,state,alpha_t,out] = tf.while_loop(while_condition, body, [total_loss,i,beta_t,state,alpha_t,out])					
			total_loss = total_loss/tf.reduce_sum(self.y_mask)
			#0.0002
		self.lr = tf.placeholder(dtype = tf.float32,shape=[])
		opt = layers.optimize_loss(loss=total_loss, learning_rate=self.lr,
				optimizer=tf.train.AdadeltaOptimizer, 
				clip_gradients = 100., global_step=self.counter_dis)

		return total_loss,opt

	def build_greedy_eval(self,max_len = 200):
		alpha_t = tf.ones([self.batch_size,self.feature_size],dtype = tf.float32)/ tf.cast(self.feature_size,tf.float32)
		beta_t = tf.zeros([self.batch_size,self.feature_size],dtype = tf.float32)

		# c = tf.matmul(self.mean_feature,self.w_init2c) + self.bias_init2c 
		state = tf.matmul(self.mean_feature,self.w_init2hid) + self.bias_init2hid
		out = state
		previous_word = tf.tile(tf.constant([111,]),[self.batch_size])
		words = []
		alphas = []
		with tf.variable_scope('Decoder'):
			for i in range(0,max_len):

				beta_t = beta_t + alpha_t
				F = tf.nn.conv2d(tf.reshape(beta_t,[-1,self.feature_height,self.feature_width,1]),self.w_B2f_filter,strides = [1,1,1,1],padding = 'SAME')
				
				#weighted x has shape [batchsize* feature size, attention dimension]
				weighted_h = tf.matmul(out, self.w_hidden) + self.bias_hidden
				weighted_h = tf.tile(weighted_h,[1,self.feature_size])
				weighted_h = tf.reshape(weighted_h,[-1,self.attention_dimension])
	 			weighted_annotation =  tf.matmul(tf.reshape(self.information_tensor,[-1,self.latent_depth]),self.w_annotation) + self.bias_annotation

				weighted_f =   tf.matmul(tf.reshape(F,[-1,self.coverage_depth]),self.w_f) + self.bias_f

				e = tf.matmul(tf.nn.tanh(tf.add(tf.add(weighted_h,weighted_annotation),weighted_f)), self.w_2e) 
				e = tf.reshape(e,[-1,self.feature_size])+ self.bias_2e

				# alpha_t = tf.nn.softmax(e)
				alpha_t = tf.exp(e)
				alpha_t = tf.multiply(alpha_t,self.vec_mask)
				alpha_t = alpha_t/tf.expand_dims(tf.reduce_sum(alpha_t,axis = -1),1)
				# expanded_alpha_t  = tf.tile(tf.expand_dims(alpha_t,2),[1,1,self.latent_depth])

				c = tf.reduce_sum(tf.multiply(tf.transpose(tf.reshape(self.information_tensor,[-1,self.feature_size,self.latent_depth]),[2,0,1]),alpha_t),axis = -1)
				c = tf.transpose(c,[1,0])


				word_embedding = tf.nn.embedding_lookup(self.w_embedding,previous_word) + self.bias_embedding
				# print('c',c.get_shape())
				# print('word_embedding',word_embedding.get_shape())
				gru_in = tf.concat([c,word_embedding],axis = 1)

				out, state = self.gru(gru_in,state)
				tf.get_variable_scope().reuse_variables()

				logit = tf.matmul(out,self.w_2logit) + self.bias_2logit
				previous_word = tf.argmax(logit,1)
				words.append(previous_word)
				alphas.append(alpha_t)
		return words,alphas

	def build_eval(self):
		self.in_alpha_t = tf.ones([self.batch_size,self.feature_size],dtype = tf.float32)/ tf.cast(self.feature_size,tf.float32)
		alpha_t = self.in_alpha_t

		self.in_beta_t = tf.zeros([self.batch_size,self.feature_size],dtype = tf.float32)
		beta_t = self.in_beta_t

		# c = tf.matmul(self.mean_feature,self.w_init2c) + self.bias_init2c 
		self.in_state = tf.matmul(self.mean_feature,self.w_init2hid) + self.bias_init2hid
		out = self.in_state
		state = self.in_state

		self.in_previous_word = tf.tile(tf.constant([111,]),[self.batch_size])
		previous_word = self.in_previous_word

		with tf.variable_scope('Decoder'):
			beta_t = beta_t + alpha_t
			F = tf.nn.conv2d(tf.reshape(beta_t,[-1,self.feature_height,self.feature_width,1]),self.w_B2f_filter,strides = [1,1,1,1],padding = 'SAME')
			
			#weighted x has shape [batchsize* feature size, attention dimension]
			weighted_h = tf.matmul(out, self.w_hidden) + self.bias_hidden
			weighted_h = tf.tile(weighted_h,[1,self.feature_size])
			weighted_h = tf.reshape(weighted_h,[-1,self.attention_dimension])
 			weighted_annotation =  tf.matmul(tf.reshape(self.information_tensor,[-1,self.latent_depth]),self.w_annotation) + self.bias_annotation

			weighted_f =   tf.matmul(tf.reshape(F,[-1,self.coverage_depth]),self.w_f) + self.bias_f

			e = tf.matmul(tf.nn.tanh(tf.add(tf.add(weighted_h,weighted_annotation),weighted_f)), self.w_2e) 
			e = tf.reshape(e,[-1,self.feature_size])+ self.bias_2e

			# alpha_t = tf.nn.softmax(e)
			alpha_t = tf.exp(e)
			alpha_t = tf.multiply(alpha_t,self.vec_mask)
			alpha_t = alpha_t/tf.expand_dims(tf.reduce_sum(alpha_t,axis = -1),1)
			# expanded_alpha_t  = tf.tile(tf.expand_dims(alpha_t,2),[1,1,self.latent_depth])

			c = tf.reduce_sum(tf.multiply(tf.transpose(tf.reshape(self.information_tensor,[-1,self.feature_size,self.latent_depth]),[2,0,1]),alpha_t),axis = -1)
			c = tf.transpose(c,[1,0])


			word_embedding = tf.nn.embedding_lookup(self.w_embedding,previous_word) + self.bias_embedding
			# print('c',c.get_shape())
			# print('word_embedding',word_embedding.get_shape())
			gru_in = tf.concat([c,word_embedding],axis = 1)

			out, state = self.gru(gru_in,state)
			tf.get_variable_scope().reuse_variables()

			logit = tf.matmul(out,self.w_2logit) + self.bias_2logit


		return alpha_t,beta_t,state,logit
