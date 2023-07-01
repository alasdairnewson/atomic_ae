# class encoder(nn.Module):

# 	for i in range(0,n_layer_conv):
# 				x = nn.Conv2d(self.channel_list[i], self.channel_list[i+1], self.kernel_size, stride=self.stride_size, padding=1)(x)
# 			self.conv_encoder = Model(input_ae,x)


import torch
import torch.nn as nn
import torch.utils as utils
from torch.autograd import Variable
import numpy as np

from scipy.interpolate import interp1d
from skimage import transform,filters

from libs.image_utils import *
from libs.data_utils import *
from libs.custom_layers import *

import pickle

model_root_dir = 'models/'
data_train_root_dir = 'data/'
#data_train_root_dir = '/home/alasdair/Alasdair/data/'
data_test_root_dir = 'data/'
results_root_dir = 'results/'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def pytorch_to_numpy(x):
	return( x.cpu().detach().numpy()  )

def pytorch_to_numpy_image(x):
	return( np.transpose(x.cpu().detach().numpy() , (0,2,3,1)).squeeze() )

def flatten(x):
	result = []
	for el in x:
		if hasattr(el, "__iter__") and not isinstance(el, str):
			result.extend(flatten(el))
		else:
			result.append(el)
	return result

def get_full_data(dataset_dir):
	# get list of files
	file_list = sorted(glob.glob(dataset_dir+'/'+"*.png"))
	if (len(file_list)==0):
		print("Error, there are no file in this directory : ",dataset_dir)
	img0 = read_image(file_list[0])
	y = np.zeros(flatten((len(file_list),img0.shape)))

	
	for i,file_name in enumerate(file_list):
		y[i,:,:] = read_image(file_name)/255.0

	y = torch.from_numpy(y).float()
	y = torch.unsqueeze(y,1) # add the channel dimension
	return y


def create_spline_image(img_size,y,x,ploy_size=3):
	img_out = np.zeros(img_size)

	spline_method = 'subsampling'

	if (spline_method == 'subsampling'):
		X = np.asarray([ [x[0]**2,x[0],1] , [x[1]**2,x[1],1] , [x[2]**2,x[2],1] ]).astype(float)
		Y = np.ravel(y).astype(float)
		a = np.dot(np.linalg.inv(X),Y)

		delta = 0.1
		sigma_blur=1.0

		x_out = np.asarray(np.arange(0,img_size[1]-1,delta))
		y_out = a[0]*np.power(x_out,2)+a[1]*x_out+a[2]

		# remove points outside the image
		x_out = x_out[ np.logical_and(y_out>=0,y_out<(img_size[0]))]
		y_out = y_out[ np.logical_and(y_out>=0,y_out<(img_size[0]))]
		img_out[np.clip(np.round(y_out),0,img_size[0]-1 ).astype(int),\
					np.clip(np.round(x_out) , 0 , img_size[1]).astype(int)] = 1
		img_out = filters.gaussian(img_out, sigma=sigma_blur)

	if (spline_method == 'super_res'):
		super_res_factor=8
		#create larger image for more finely sampled spline
		x_temp = super_res_factor*x
		y_temp = super_res_factor*y
		X = np.asarray([ [x_temp[0]**2,x_temp[0],1] , [x_temp[1]**2,x_temp[1],1] , [x_temp[2]**2,x_temp[2],1] ]).astype(float)
		Y = np.ravel(y_temp).astype(float)
		a = np.dot(np.linalg.inv(X),Y)

		x_out = np.asarray(range(0,super_res_factor*img_size[1]))
		y_out = a[0]*np.power(x_out,2)+a[1]*x_out+a[2]
		img_out_super_res = np.zeros((super_res_factor*img_size[0],super_res_factor*img_size[1]))

		# remove points outside the image
		x_out = x_out[ np.logical_and(y_out>=0,y_out<(super_res_factor*img_size[0]))]
		y_out = y_out[ np.logical_and(y_out>=0,y_out<(super_res_factor*img_size[0]))]
		img_out_super_res[np.clip(np.round(y_out),0,super_res_factor*img_size[0]-1 ).astype(int),\
					np.clip(np.round(x_out) , 0 , super_res_factor*img_size[1]).astype(int)] = 1

		img_out = transform.resize(img_out_super_res,img_size,anti_aliasing=True)
	elif(spline_method == 'monte_carlo'):
		n_monte_carlo = 100
		sigma_monte_carlo = 0.8
		eps_vect = sigma_monte_carlo*np.random.randn(n_monte_carlo,poly_size)

		for i in range(0,n_monte_carlo):
			x_temp = x+eps_vect[i,:].reshape((poly_size,1))
			y_temp = y+eps_vect[i,:].reshape((poly_size,1))
			X = np.asarray([ [x_temp[0]**2,x_temp[0],1] , [x_temp[1]**2,x_temp[1],1] , [x_temp[2]**2,x_temp[2],1] ]).astype(float)
			Y = np.ravel(y_temp).astype(float)
			a = np.dot(np.linalg.inv(X),Y)

			x_out = np.asarray(range(0,img_size[1]))
			y_out = a[0]*np.power(x_out,2)+a[1]*x_out+a[2]

			# remove points outside the image
			x_out = x_out[ np.logical_and(y_out>=0,y_out<img_size[0])]
			y_out = y_out[ np.logical_and(y_out>=0,y_out<img_size[0])]
			img_out[np.clip(np.round(y_out),0,img_size[0]-1 ).astype(int),np.clip(np.round(x_out) , 0 , img_size[1]).astype(int)] = \
				img_out[np.clip(np.round(y_out),0,img_size[0]-1 ).astype(int),np.clip(np.round(x_out) , 0 , img_size[1]).astype(int)] +1
		
		img_out /= n_monte_carlo

	# normalise image to (0,1)
	img_out = img_out/img_out.max()

	return(img_out,a)

def get_spline_data_batch(batch_size,img_size,poly_size=3,is_train=1,shuffle_samples=1):
	p_y = np.zeros((poly_size,batch_size))
	p_x = np.zeros((poly_size,batch_size))

	p_y = (img_size[0]-1)*np.random.random((poly_size,batch_size))

	# p_y[0,:] = np.random.randint(0,int(img_size[0]/2.0),(1,batch_size))
	# p_y[1,:] = np.random.randint(int(img_size[0]/2.0),img_size[0],(1,batch_size))
	# p_y[2,:] = np.random.randint(0,int(img_size[0]/2.0),(1,batch_size))

	p_x[0,:] = (img_size[1]/float(poly_size)-1)*np.random.random((1,batch_size))
	p_x[1,:] = img_size[1]/float(poly_size) + (img_size[1]/float(poly_size)-1)*np.random.random((1,batch_size))
	p_x[2,:] = 2*img_size[1]/float(poly_size) + (img_size[1]/float(poly_size)-1)*np.random.random((1,batch_size))


	y = np.zeros((batch_size,img_size[0],img_size[1]))
	theta = np.zeros((poly_size,batch_size))
	for i in range(0,batch_size):
		y[i,:,:],theta[:,i] = create_spline_image(img_size,np.reshape(p_y[:,i],(poly_size,1)),np.reshape(p_x[:,i],(poly_size,1)))


def get_data_batch(dataset_dir,batch_size,shuffle_samples=1):

	theta = 0
	# get list of files
	file_list = sorted(glob.glob(dataset_dir+"*.png"))
	if (len(file_list)==0):
		print("Error, there are no file in this directory : ",dataset_dir)
	img0 = read_image(file_list[0])
	y = np.zeros((batch_size,img0.shape[0],img0.shape[1],img0.shape[2]))

	file_numbers = np.asarray(range(0,len(file_list)))
	if (shuffle_samples>0):
		if (shuffle_samples>1):
			np.random.seed(shuffle_samples)
		np.random.shuffle(file_numbers)
	batch_list = file_numbers[0:batch_size]

	for i in range(0,batch_size):
		y[i,:,:,:] = read_image(file_list[batch_list[i]])

	y = torch.from_numpy( np.transpose(y,(0,3,1,2))).float()
	return y,theta

def image_to_mosaic(self,x,block_size):
	img_shape = x.shape
	n_blocks = (np.asarray(img_shape[2:4])/np.asarray(block_size)).astype(int)
	# k = 0
	img_mosaic = x[ :, 0, 0 : block_size[0] , 0 : block_size[1]]
	for i in range(0,n_blocks[0]):
		for j in range(0,n_blocks[1]):
			if (i != 0 or j != 0):
				img_mosaic = torch.cat( (img_mosaic,x[:,0, (i*block_size[0]) : ( (i+1)*block_size[0]) , \
												(j*block_size[1]) : ( (j+1)*block_size[1])]), axis=1)

	return img_mosaic


def mosaic_to_image(z,z_size,n_blocks,decoder_block):	
	k = 0
	line_out = decoder_block(z[ :, :, k : (k+1)*z_size ])

	k = k+1
	for j in range(1,n_blocks[1]):
		img_block = decoder_block(z[:, :, k*z_size : (k+1)*z_size])
		line_out = torch.cat( (line_out,img_block), axis=3)
		k=k+1
	img_out = line_out

	for i in range(1,n_blocks[0]):
		line_out = decoder_block(z[ :, : , k*z_size : (k+1)*z_size ])
		k=k+1
		for j in range(1,n_blocks[1]):
			img_block = decoder_block(z[:, : , k*z_size : (k+1)*z_size])
			line_out = torch.cat( (line_out,img_block), axis=3)
			k=k+1
		img_out = torch.cat((img_out,line_out),axis=2)
	return img_out


class autoencoder(nn.Module):
	def __init__(self, model_id='', toggle_save_model = 1, training_object = 'spline',data_train_dir='',lambda_z=1.0):
		super(autoencoder, self).__init__()

		# parameters
		self.root_dir = ""
		self.training_object = training_object
		self.img_size = (128,128)#(256,256)
		self.block_size = (32,32)
		if (self.training_object == 'spikes_repulsive_all_k'):
			self.n_blocks = (10,1)#(10,1)
		else:
			self.n_blocks = (np.asarray(self.img_size)/np.asarray(self.block_size)).astype(int)
		self.kernel_size = (3,3)
		self.stride_size = (2,2)
		self.poly_size = 6
		if (self.training_object == 'spikes_repulsive_all_k'):
			self.z_size = 12#6#3
		else:
			self.z_size = 6#
		self.z_size_full = self.z_size * (np.prod(self.n_blocks))
		self.z_spatial_size = 2
		self.lambda_z = 10e-6#lambda_z
		self.use_bias = True
		self.alpha = 0.2
		self.batch_size = 64
		self.n_epochs = 1000000
		self.toggle_save_model = toggle_save_model
		if(self.training_object == 'spikes_repulsive_all_k'):
			self.learning_rate = 0.0001#
		else:
			self.learning_rate = 0.0005
		self.loss_list = []
		if (data_train_dir == ''):	# default directory
			self.data_train_dir = data_train_root_dir+self.training_object+'_train/'
		else:	# specify directory
			self.data_train_dir = data_train_dir
		self.data_test_dir = data_test_root_dir+self.training_object+'_test/'
		print("data_train_dir : ",self.data_train_dir)

		self.model_root_dir = model_root_dir
		self.results_root_dir = results_root_dir
		self.model_id = model_id
		self.n_filters = 16

		[X,Y] = get_data_batch(self.data_train_dir,batch_size=1,shuffle_samples=0)

		self.n_channels_in = X.shape[1]
		self.channel_list = np.asarray([self.n_channels_in,32,16,8,8,8,8])

		if(self.training_object == 'mnist'):
			self.architecture = 'conv_mlp_sum'
		else:
			self.architecture = 'conv_mlp_sum'
		
		if (self.architecture == 'conv_mlp'):
			n_layer_conv = int(np.log2(np.minimum(self.img_size[0],self.img_size[1])/self.z_spatial_size))
			self.z_channel_size = self.channel_list[n_layer_conv]
		elif(self.architecture == 'conv_mlp_sum' or self.architecture == 'conv_mlp_max'):
			n_layer_conv = int(np.log2(np.minimum(self.block_size[0],self.block_size[1])/self.z_spatial_size))
			self.z_channel_size_encoder = self.channel_list[n_layer_conv]
			n_layer_conv = int(np.log2(np.minimum(self.img_size[0],self.img_size[1])/self.z_spatial_size))
			self.z_channel_size_decoder = self.channel_list[n_layer_conv]
		

		print('Training object : ',self.training_object)
		print('Architecture : ',self.architecture)

		# in case we have to actually create the necessary models and results directories
		if (os.path.exists(self.root_dir+self.model_root_dir) == 0):
			create_directory(self.root_dir+self.model_root_dir)
		if (os.path.exists(self.root_dir+self.results_root_dir) == 0):
			create_directory(self.root_dir+self.results_root_dir)

		if (model_id =='' and self.toggle_save_model >0 ):
			#create model directory
			self.model_id = create_new_param_id()
			self.param_dir = self.root_dir+self.model_root_dir+self.training_object+'/'+self.model_id
			# if the specific training object model directory does not exist
			if (os.path.exists(self.root_dir+self.model_root_dir+self.training_object) == 0):
				create_directory(self.root_dir+self.model_root_dir+self.training_object)
			# create the directory
			os.mkdir(self.param_dir)
		# now deal with the results directory, which potentially might not exist
		self.results_dir = self.root_dir+self.results_root_dir+self.training_object+'/'+self.model_id+'/'
		if (os.path.exists(self.results_dir) == 0):
			#create the results directory if necessary
			create_directory(self.results_dir)

		# create the model itself
		x_size = np.prod(self.img_size)

		if (self.architecture=='mlp'):
			coef_reduc = 64
			self.encoder = nn.Sequential(
				nn.Linear(x_size,int(x_size/coef_reduc),bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True),
				nn.Linear(int(x_size/coef_reduc),self.z_size,bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True)
			)
			self.decoder = nn.Sequential(
				nn.Linear(self.z_size, int(x_size/coef_reduc),bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True),
				nn.Linear(int(x_size/coef_reduc), x_size,bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True)
			)
		elif(self.architecture=='convolutional'):
			self.encoder = nn.Sequential(
				nn.Conv2d(1, self.n_filters, self.kernel_size, stride=self.stride_size, padding=1), 
				nn.LeakyReLU(negative_slope=self.alpha), #nn.Sigmoid(), #
				nn.Conv2d(self.n_filters, 16, self.kernel_size, stride=self.stride_size, padding=1),  
				nn.LeakyReLU(negative_slope=self.alpha), #nn.Sigmoid(), #
				nn.Conv2d(16, 8, self.kernel_size, stride=self.stride_size, padding=1),  
				nn.LeakyReLU(negative_slope=self.alpha), #nn.Sigmoid(), #
				nn.Conv2d(8, 4, self.kernel_size, stride=self.stride_size, padding=1),  
				nn.LeakyReLU(negative_slope=self.alpha), #nn.Sigmoid(), #
				nn.Conv2d(4, 4, self.kernel_size, stride=self.stride_size, padding=1),  
				nn.LeakyReLU(negative_slope=self.alpha), #nn.Sigmoid(), #
				nn.Conv2d(4, 4, self.kernel_size, stride=self.stride_size, padding=1),  
				nn.LeakyReLU(negative_slope=self.alpha), #nn.Sigmoid(), #
				nn.Conv2d(4, 4, self.kernel_size, stride=self.stride_size, padding=1), 
				nn.LeakyReLU(True), #nn.Sigmoid(), #
			)
			self.decoder = nn.Sequential(
				nn.Conv2d(4, 4, self.kernel_size, stride=(1,1),padding=1), 
				nn.Upsample(scale_factor=2),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True), #nn.Sigmoid(), #
				nn.Conv2d(4, 4, self.kernel_size, stride=(1,1), padding=1),  
				nn.Upsample(scale_factor=2),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True), #nn.Sigmoid(), #
				nn.Conv2d(4, 4, self.kernel_size, stride=(1,1), padding=1),  
				nn.Upsample(scale_factor=2),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True), #nn.Sigmoid(), #
				nn.Conv2d(4, 8, self.kernel_size, stride=(1,1), padding=1),  
				nn.Upsample(scale_factor=2),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True), #nn.Sigmoid(), #
				nn.Conv2d(8, 16, self.kernel_size, stride=(1,1), padding=1), 
				nn.Upsample(scale_factor=2),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True), #nn.Sigmoid(), #
				nn.Conv2d(16, self.n_filters, self.kernel_size, stride=(1,1), padding=1), 
				nn.Upsample(scale_factor=2),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True), #nn.Sigmoid(), #
				nn.Conv2d(self.n_filters, 1, self.kernel_size, stride=(1,1), padding=1), 
				nn.Upsample(scale_factor=2),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True), #nn.Sigmoid(), #
			)
		elif(self.architecture=='conv_mlp'):
			self.z_conv_size = self.z_spatial_size*self.z_spatial_size*self.z_channel_size

			# project to space of size z_conv_size*z_conv_size*z_channel_size
			n_layer_conv = int(np.log2(np.minimum(self.img_size[0],self.img_size[1])/self.z_spatial_size))
			conv_modules = []
			for i in range(0,n_layer_conv):
				conv_modules.append(nn.Conv2d(self.channel_list[i], self.channel_list[i+1], self.kernel_size, stride=self.stride_size, padding=1))
				conv_modules.append(nn.LeakyReLU(negative_slope=self.alpha))
			self.conv_encoder = nn.Sequential(*conv_modules)
			
			self.mlp_encoder = nn.Sequential(
				nn.Linear(self.z_conv_size,int(self.z_conv_size),bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True),
				nn.Linear(self.z_conv_size,self.z_size,bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True)
			)
			self.encoder =  nn.Sequential(
				self.conv_encoder,
				View((-1,self.z_conv_size)),
				self.mlp_encoder
			)
			self.mlp_decoder = nn.Sequential(
				nn.Linear(self.z_size, self.z_conv_size,bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True),
				nn.Linear(self.z_conv_size, self.z_conv_size,bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True)
			)
			deconv_modules = []
			for i in np.arange(n_layer_conv,0,-1):
				deconv_modules.append(nn.Conv2d(self.channel_list[i], self.channel_list[i-1], self.kernel_size, stride=1, padding=1))
				deconv_modules.append(nn.Upsample(scale_factor=2))
				deconv_modules.append(nn.LeakyReLU(negative_slope=self.alpha,inplace=True))
			self.conv_decoder = nn.Sequential(*deconv_modules)

			self.decoder =  nn.Sequential(
				self.mlp_decoder,
				View((self.z_channel_size,self.z_spatial_size,self.z_spatial_size)),
				self.conv_decoder
			)
		elif(self.architecture=='conv_mlp_sum'):
			
			# ENCODER 
			n_layer_conv = int(np.log2(np.minimum(self.block_size[0],self.block_size[1])/self.z_spatial_size))
			# project to space of n_blocks*z_spatial_size*z_spatial_size*z_channel_size
			self.z_conv_size_full = \
			int(np.prod( (self.img_size[0]/( (self.stride_size[0])**n_layer_conv),self.img_size[1]/( (self.stride_size[1])**n_layer_conv)) ) * self.z_channel_size_encoder)

			conv_modules = []
			for i in range(0,n_layer_conv):
				conv_modules.append(nn.Conv2d(self.channel_list[i], self.channel_list[i+1], self.kernel_size, stride=self.stride_size, padding=1))
				conv_modules.append(nn.LeakyReLU(negative_slope=self.alpha))
				#conv_modules.append(nn.BatchNorm2d(self.channel_list[i+1]))
			self.conv_encoder = nn.Sequential(*conv_modules)

			self.mlp_encoder = nn.Sequential(
				nn.Linear(self.z_conv_size_full,int(self.z_conv_size_full),bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True),
				#nn.BatchNorm1d(1),
				nn.Linear(self.z_conv_size_full,self.z_size_full,bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True)#,
				#nn.BatchNorm1d(1)
			)
			self.encoder =  nn.Sequential(
					self.conv_encoder,
					View((-1,self.z_conv_size_full)),
					self.mlp_encoder
				)
			# if (self.training_object == 'spikes_repulsive_all_k'):
			# 	self.encoder =  nn.Sequential(
			# 		self.conv_encoder,
			# 		View((-1,self.z_conv_size_full)),
			# 		self.mlp_encoder,
			# 		Sort_Latent_Layer(self.z_size),
			# 	)
			# else:
			# 	self.encoder =  nn.Sequential(
			# 		self.conv_encoder,
			# 		View((-1,self.z_conv_size_full)),
			# 		self.mlp_encoder
			# 	)

			# DECODER
			# now create unitary decoder
			self.z_conv_size = self.z_spatial_size*self.z_spatial_size*self.z_channel_size_decoder
			# project from space of size z_conv_size*z_conv_size*z_channel_size to img_size
			n_layer_conv = int(np.log2(np.minimum(self.img_size[0],self.img_size[1])/self.z_spatial_size))
				
			self.mlp_decoder = nn.Sequential(
				nn.Linear(self.z_size, self.z_conv_size,bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True),
				#nn.BatchNorm1d(1),
				nn.Linear(self.z_conv_size, self.z_conv_size,bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True)#,
				#nn.BatchNorm1d(1)
			)
			deconv_modules = []
			for i in np.arange(n_layer_conv,0,-1):
				deconv_modules.append(nn.Conv2d(self.channel_list[i], self.channel_list[i-1], self.kernel_size, stride=1, padding=1))
				deconv_modules.append(nn.Upsample(scale_factor=2))
				deconv_modules.append(nn.LeakyReLU(negative_slope=self.alpha))
				deconv_modules.append(nn.Conv2d(self.channel_list[i-1], self.channel_list[i-1], self.kernel_size, stride=1, padding=1))
				deconv_modules.append(nn.LeakyReLU(negative_slope=self.alpha))
				#deconv_modules.append(nn.BatchNorm2d(self.channel_list[i-1]))
			# get rid of last BatchNorm
			#deconv_modules.pop()
			self.conv_decoder = nn.Sequential(*deconv_modules)

			self.decoder_block =  nn.Sequential(
					self.mlp_decoder,
					View((self.z_channel_size_decoder,self.z_spatial_size,self.z_spatial_size)),
					self.conv_decoder
				)
			self.decoder =  nn.Sequential(
				Decoder_Sum_Layer(self.z_size, self.n_blocks, self.decoder_block)
			)
		elif(self.architecture=='conv_mlp_max'):
			
			# ENCODER 
			n_layer_conv = int(np.log2(np.minimum(self.block_size[0],self.block_size[1])/self.z_spatial_size))
			# project to space of n_blocks*z_spatial_size*z_spatial_size*z_channel_size
			self.z_conv_size_full = \
			int(np.prod( (self.img_size[0]/( (self.stride_size[0])**n_layer_conv),self.img_size[1]/( (self.stride_size[1])**n_layer_conv)) ) * self.z_channel_size_encoder)

			conv_modules = []
			for i in range(0,n_layer_conv):
				conv_modules.append(nn.Conv2d(self.channel_list[i], self.channel_list[i+1], self.kernel_size, stride=self.stride_size, padding=1))
				conv_modules.append(nn.LeakyReLU(negative_slope=self.alpha))
				conv_modules.append(nn.BatchNorm2d(self.channel_list[i+1]))
			self.conv_encoder = nn.Sequential(*conv_modules)

			self.mlp_encoder = nn.Sequential(
				nn.Linear(self.z_conv_size_full,int(self.z_conv_size_full),bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True),
				#nn.BatchNorm1d(1),
				nn.Linear(self.z_conv_size_full,self.z_size_full,bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True)#,
				#nn.BatchNorm1d(1)
			)
			self.encoder =  nn.Sequential(
				self.conv_encoder,
				View((-1,self.z_conv_size_full)),
				self.mlp_encoder
			)

			# DECODER
			# now create unitary decoder
			self.z_conv_size = self.z_spatial_size*self.z_spatial_size*self.z_channel_size_decoder
			# project from space of size z_conv_size*z_conv_size*z_channel_size to img_size
			n_layer_conv = int(np.log2(np.minimum(self.img_size[0],self.img_size[1])/self.z_spatial_size))
				
			self.mlp_decoder = nn.Sequential(
				nn.Linear(self.z_size, self.z_conv_size,bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True),
				#nn.BatchNorm1d(1),
				nn.Linear(self.z_conv_size, self.z_conv_size,bias=self.use_bias),
				nn.LeakyReLU(negative_slope=self.alpha,inplace=True)#,
				#nn.BatchNorm1d(1)
			)
			deconv_modules = []
			for i in np.arange(n_layer_conv,0,-1):
				deconv_modules.append(nn.Conv2d(self.channel_list[i], self.channel_list[i-1], self.kernel_size, stride=1, padding=1))
				deconv_modules.append(nn.Upsample(scale_factor=2))
				deconv_modules.append(nn.LeakyReLU(negative_slope=self.alpha))
				#deconv_modules.append(nn.BatchNorm2d(self.channel_list[i-1]))
			# get rid of last BatchNorm
			#deconv_modules.pop()
			self.conv_decoder = nn.Sequential(*deconv_modules)

			self.decoder_block =  nn.Sequential(
					self.mlp_decoder,
					View((self.z_channel_size_decoder,self.z_spatial_size,self.z_spatial_size)),
					self.conv_decoder
				)
			self.decoder =  nn.Sequential(
				Decoder_Max_Layer(self.z_size, self.n_blocks, self.decoder_block)
			)
		elif(self.architecture=='conv_mlp_block'):
				self.z_conv_size_full = np.prod(self.n_blocks)*self.z_spatial_size*self.z_spatial_size*self.z_channel_size
				
				# project to space of size z_conv_size*z_conv_size*z_channel_size
				n_layer_conv = int(np.log2(np.minimum(self.block_size[0],self.block_size[1])/self.z_spatial_size))
				conv_modules = []
				for i in range(0,n_layer_conv):
					conv_modules.append(nn.Conv2d(self.channel_list[i], self.channel_list[i+1], self.kernel_size, stride=self.stride_size, padding=1))
				self.conv_encoder = nn.Sequential(*conv_modules)

				self.mlp_encoder = nn.Sequential(
					nn.Linear(self.z_conv_size_full,int(self.z_conv_size_full),bias=self.use_bias),
					nn.LeakyReLU(negative_slope=self.alpha,inplace=True),
					nn.Linear(self.z_conv_size_full,self.z_size_full,bias=self.use_bias),
					nn.LeakyReLU(negative_slope=self.alpha,inplace=True)
				)
				self.encoder =  nn.Sequential(
					self.conv_encoder,
					View((-1,self.z_conv_size_full)),
					self.mlp_encoder
				)
				# now create unitary decoder
				self.z_conv_size = self.z_spatial_size*self.z_spatial_size*self.z_channel_size
				
				self.mlp_decoder = nn.Sequential(
					nn.Linear(self.z_size, self.z_conv_size,bias=self.use_bias),
					nn.LeakyReLU(negative_slope=self.alpha,inplace=True),
					nn.Linear(self.z_conv_size, self.z_conv_size,bias=self.use_bias),
					nn.LeakyReLU(negative_slope=self.alpha,inplace=True)
				)
				deconv_modules = []
				for i in np.arange(n_layer_conv,0,-1):
					deconv_modules.append(nn.Conv2d(self.channel_list[i], self.channel_list[i-1], self.kernel_size, stride=1, padding=1))
					deconv_modules.append(nn.Upsample(scale_factor=2))
				self.conv_decoder = nn.Sequential(*deconv_modules)

				self.decoder_block =  nn.Sequential(
					self.mlp_decoder,
					View((self.z_channel_size,self.z_spatial_size,self.z_spatial_size)),
					self.conv_decoder
				)
				self.decoder =  nn.Sequential(
					Mosaic(self.z_size, self.n_blocks, self.decoder_block)
				)



	def forward(self, x):
		if (self.architecture=='mlp'):
			x = x.view((-1,np.prod(self.img_size)))
		if (self.architecture=='mlp' or self.architecture=='convolutional'):
			x = self.encoder(x)
			x = self.decoder(x)
		elif (self.architecture=='conv_mlp'):
			x = self.encoder(x)
			x = self.decoder(x)
		elif (self.architecture=='conv_mlp_sum'):
			z_full = self.encoder(x)
			# create mosaic list
			x = Decoder_Sum_Layer(self.z_size,self.n_blocks, self.decoder_block)(z_full)
		elif (self.architecture=='conv_mlp_max'):
			z_full = self.encoder(x)
			# create mosaic list
			x = Decoder_Max_Layer(self.z_size,self.n_blocks, self.decoder_block)(z_full)
		elif (self.architecture=='conv_mlp_block'):
			z_full = self.encoder(x)
			# create mosaic list
			x = mosaic_to_image(z_full, self.z_size,self.n_blocks,self.decoder_block)
		if (self.architecture=='mlp'):
			x = x.view((-1,1,self.img_size[0],self.img_size[1]))
		return x

	def print_shapes(self, x):
		print("Input :")
		print(x.shape)
		print("Encoder :")
		if (self.architecture=='mlp'):
			x = x.flatten()
		for layer in self.encoder:
			x = layer(x)
			print(x.size())

		print("Decoder :")
		for layer in self.decoder:
			x = layer(x)
			print(x.size())
		if (self.architecture=='mlp'):
			x = x.view((-1,1,self.img_size[0],self.img_size[1]))
		print("Output :")
		print(self.forward(x).shape)
		return

	def save_model(self,model_file):
		torch.save(self, model_file)#torch.save(self.state_dict(), model_file)

	def z_group_sparsity(self,z):

		z_gs_out = torch.sqrt( (z[ :, :, 0::self.z_size ]**2).sum(axis=2,keepdims=True))

		for i in range(1,self.z_size):
			z_gs_out = z_gs_out+torch.sqrt( (z[ :, :, i::self.z_size ]**2).sum(axis=2,keepdims=True))
		z_gs_out = z_gs_out.mean()
		return z_gs_out

	def loss_fn_custom(self,y, x):

		z = self.encoder(x.to(device))
		#y_prod_loss = Decoder_Prod_Layer(self.z_size, self.n_blocks, self.decoder_block)(z)**2
		mse_loss = torch.nn.MSELoss(reduction='mean')(y,x)

		z_loss = self.z_group_sparsity(z)

		loss = mse_loss+self.lambda_z*z_loss#torch.mean(mse_loss)#+y_prod_loss)
		loss_full = (loss,mse_loss,z_loss)

		return loss_full

	def plot_loss(self):
		loss_list_temp = np.asarray(self.loss_list)
		if (loss_list_temp.ndim==2):
			for i in range(1,loss_list_temp.shape[1]):
				plt.plot(loss_list_temp[:,i])
		else:
			plt.plot(loss_list_temp)

		plt.legend(['mse', 'z_sparsity'])
		plt.show()

	def plot_autoencoding(self,output_dir='',input_output=0):
		n_test = 5
		n_batch = 3

		m = int(np.round(np.sqrt(n_test)))
		n = int(np.ceil(float(n_test)/float(m)))

		theta_list = np.linspace(0,2*np.pi/3,n_test)
		_r = self.img_size[0]/4.0

		# y = np.zeros(flatten( (len(theta_list),1,self.img_size) ))
		# sigma = 0.8
		# n_monte_carlo = 400
		# for i,theta in enumerate(theta_list):
		# 	params = np.asarray( (_r,theta))
		# 	y[i,0,:,:] = monte_carlo_shape(self.img_size,self.training_object,params,sigma,n_monte_carlo)

		if(hasattr(self,'data_test_dir')):
			# mnist : 10
			y,_ = get_data_batch(self.data_test_dir,batch_size=self.batch_size,shuffle_samples=5)
		else:
			y,_ = get_data_batch(data_test_root_dir+self.training_object+'_test/',batch_size=self.batch_size,shuffle_samples=1)

		fig, axs = plt.subplots(2,n_test)
		for i in range(0,n_test):
			y_in = torch.unsqueeze(y[i+n_batch*n_test,:,:,:],axis=0)
			y_out = pytorch_to_numpy_image( self.forward(y_in.to(device)) )
			y_in = pytorch_to_numpy_image(y_in)
			y_out = np.clip( y_out,0.0,1.0) 
			# y_out = (y_out-y_out.min())
			# y_out = y_out/y_out.max()
			
			# y_out[y_in>0.0,:] = 0
			# y_out[y_in>0.0,1] = 1.0

			#y_out[y_in==1] = 1.0
			# if we want to save all the input/output images individually
			if (output_dir != '' and input_output>0):
				imageio.imwrite(output_dir+'auto_encoding_img_'+str(i).zfill(3)+'_input.png',(255.*y_in).astype(np.uint8))
				imageio.imwrite(output_dir+'auto_encoding_img_'+str(i).zfill(3)+'_output.png',(255.*y_out).astype(np.uint8))
			axs[0,i].imshow( (y_in),cmap="gray")
			axs[0,i].set_xticks([])
			axs[0,i].set_yticks([])
			axs[1,i].imshow( (y_out),cmap="gray")
			axs[1,i].set_xticks([])
			axs[1,i].set_yticks([])
			#axs[i,j].imshow(y_out.data.numpy().squeeze(),cmap='gray')
			#axs[i,j].title.set_text('z : '+str(self.encoder(y_in).data.numpy().squeeze()))
		plt.tight_layout()
		if (output_dir != ''):
			plt.savefig(output_dir+'auto_encoding.png')
		else:
			plt.show()



		# z = self.encoder(torch.reshape( torch.tensor(y).float(),(y.shape[0],1,self.img_size[0],self.img_size[1]))).data.numpy().squeeze()
		# plt.plot(theta_list,z)
		# plt.xlabel('Rotation')
		# plt.ylabel('z')
		# plt.title('Number of filters : '+str(self.n_filters))

		# plt.show()

	def plot_autoencoding_stability(self):
		n_test = 5
		n_stability = 3

		# y = np.zeros(flatten( (len(theta_list),1,self.img_size) ))
		# sigma = 0.8
		# n_monte_carlo = 400
		# for i,theta in enumerate(theta_list):
		# 	params = np.asarray( (_r,theta))
		# 	y[i,0,:,:] = monte_carlo_shape(self.img_size,self.training_object,params,sigma,n_monte_carlo)

		if(hasattr(self,'data_test_dir')):
			y,_ = get_data_batch(self.data_test_dir,batch_size=self.batch_size,shuffle_samples=0)
		else:
			y,_ = get_data_batch(data_test_root_dir+self.training_object+'_test/',batch_size=self.batch_size,shuffle_samples=0)
		block_img_list= []

		fig, axs = plt.subplots(4,n_test)
		y_init = torch.unsqueeze(y[1,:,:,:],axis=0)
		for i in range(0,n_test):
			y_in = y_init.clone()
			z_in = torch.squeeze(self.encoder(y_in)[:,:,(i*self.z_size):((i+1)*self.z_size)],axis=1).to(device)
			y_in = self.decoder_block(z_in)
			for j in range(0,n_stability):
				z_in = torch.squeeze(self.encoder(self.decoder_block(z_in))[:,:,(i*self.z_size):((i+1)*self.z_size)],axis=1)
			y_out = self.decoder_block(z_in)

			y_out = pytorch_to_numpy_image( y_out )
			y_in = pytorch_to_numpy_image(y_in)
			y_out = (y_out-y_out.min())
			y_out = y_out/y_out.max()
			
			y_in = (y_in-y_in.min())
			y_in = y_in/y_in.max()

			axs[3,i].plot( pytorch_to_numpy(torch.squeeze(z_in)))
			if (y_in.ndim == 2):
				axs[0,i].imshow( (y_in),cmap="gray")
			else:
				axs[0,i].imshow( (y_in))
			z_in = torch.squeeze(self.encoder(y_init)[:,:,(i*self.z_size):((i+1)*self.z_size)])
			axs[1,i].plot( pytorch_to_numpy(z_in))
			if (y_out.ndim == 2):
				axs[2,i].imshow( (y_out),cmap="gray")
			else:
				axs[2,i].imshow( (y_out))
			#axs[i,j].imshow(y_out.data.numpy().squeeze(),cmap='gray')
			#axs[i,j].title.set_text('z : '+str(self.encoder(y_in).data.numpy().squeeze()))
		plt.show()



	def show_decoder(self,output_dir=''):


		if(hasattr(self,'data_train_dir')):
			x,_ = get_data_batch(self.data_test_dir,batch_size=128,shuffle_samples=0)
		else:
			x,_  = get_data_batch(data_test_root_dir+self.training_object+'_test/',batch_size=128,shuffle_samples=0)

		z = self.encoder(x.to(device))

		delta = 0.25
		self.z_block_display = 4
		sample_index = 100#0#10#39#51
		img_in = torch.unsqueeze(x[sample_index,:,:,:],axis=0)
		z_0 = self.encoder(img_in.to(device))

		min_z = torch.min(z[:,:,self.z_size*self.z_block_display:(self.z_size*(self.z_block_display+1))],axis=0)[0]
		max_z = torch.max(z[:,:,self.z_size*self.z_block_display:(self.z_size*(self.z_block_display+1))],axis=0)[0]

		if (self.architecture=='mlp'):
			y = pytorch_to_numpy_image( self.decoder( torch.reshape(torch.tensor(z_0).float(),(-1,self.z_size)).to(device) ))
			y = np.reshape(y,(self.img_size))
		elif(self.architecture=='convolutional'):
			y = pytorch_to_numpy_image( self.decoder( torch.reshape(torch.tensor(z_0).float(),(-1,self.z_size,1,1)).to(device) ))
		elif(self.architecture=='conv_mlp'):
			y = pytorch_to_numpy_image( self.decoder( torch.reshape(torch.tensor(z_0).float(),(-1,self.z_size)).to(device) ))
		elif(self.architecture=='conv_mlp_sum' or self.architecture=='conv_mlp_max'):
			y = pytorch_to_numpy_image(self.decoder(torch.tensor(z_0).float().to(device) ))

		fig, ax = plt.subplots()
		#fig = plt.figure()
		plt.subplots_adjust(left=0.25, bottom=0.25)
			
		#show result
		curr_ax = ax.imshow(y,cmap='gray')

		axcolor = 'lightgoldenrodyellow'
		axs = []
		slider = []

		for i in range(0,self.z_size):
			axs.append(plt.axes([0.25, 0.1+0.05*i, 0.65, 0.03], facecolor=axcolor))
			slider.append(Slider(axs[i], 'z'+str(i), float(min_z[0,i]), float(max_z[0,i]), valinit=float(z_0[0,0,self.z_size*self.z_block_display+i])))

		# set up block choice
		ax_box = fig.add_axes([0.1, 0.4, 0.1, 0.075])
		text_box = TextBox(ax_box, "Block number")#, verticalalignment="center")


		def update_plot(z_update):
			#plot data
			if (self.architecture=='mlp'):
				y = pytorch_to_numpy_image( self.decoder( torch.reshape(torch.tensor(z_update).float(),(-1,self.z_size)) ))
				y = np.reshape(y,(self.img_size))
			elif(self.architecture=='convolutional'):
				y = pytorch_to_numpy_image( self.decoder( torch.reshape(torch.tensor( ).float(),(-1,self.z_size,1,1)) ))
			elif(self.architecture=='conv_mlp'):
				y = pytorch_to_numpy_image( self.decoder( torch.reshape(torch.tensor(z_update).float(),(-1,self.z_size)) ))
			elif(self.architecture=='conv_mlp_sum' or self.architecture=='conv_mlp_max'):
				y = pytorch_to_numpy_image(self.decoder( z_update.to(device) ))
			ax.imshow(y,cmap='gray')
			# axs.relim()
			# axs.autoscale_view()
			#l.set_data(y)
			fig.canvas.draw()

		def update_text_box(expression):
			self.z_block_display = int(eval(expression))
			print("z block display : ",self.z_block_display)
			z_temp = z_0.clone()
			min_z = torch.min(z[:,:,self.z_block_display*self.z_size:((self.z_block_display+1)*self.z_size)],axis=0)[0]
			max_z = torch.max(z[:,:,self.z_block_display*self.z_size:((self.z_block_display+1)*self.z_size)],axis=0)[0]

			for i in range(0,self.z_size):
				z_temp[0,0,i+self.z_block_display*self.z_size] = slider[i].val
				slider[i].reset()
				slider[i].valmin = float(min_z[0,i])
				slider[i].valmax = float(max_z[0,i])
				slider[i].valinit = float(z_0[:,0,i+self.z_block_display*self.z_size])

				print("min val : ",slider[i].valmin)
				print("max val : ",slider[i].valmax)
				print("val init: ",slider[i].valinit)

				slider[i].ax.set_xlim(slider[i].valmin,slider[i].valmax)
				slider[i].set_val(slider[i].valinit)

			update_plot(z_temp)

		def update_slider(val):
			z_temp = z_0.clone()
			for i in range(self.z_block_display*self.z_size,self.z_block_display*self.z_size+self.z_size):
				z_temp[0,0,i] = slider[i-self.z_block_display*self.z_size].val
			update_plot(z_temp)

		text_box.on_submit(update_text_box)
		text_box.set_val(str(self.z_block_display))

		for i in range(0,self.z_size):
			slider[i].on_changed(update_slider)

		resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
		button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

		def reset(event):
			slider.reset()
			text_box.set_val("0")


		button.on_clicked(reset)

		plt.show()

	def show_block_decoder(self,output_dir=''):

		if(hasattr(self,'data_test_dir')):
			x,_ = get_data_batch(self.data_test_dir,batch_size=self.batch_size,shuffle_samples=0)
		else:
			x,_ = get_data_batch(data_test_root_dir+self.training_object+'_test/',batch_size=self.batch_size,shuffle_samples=0)

		m = int(np.ceil(np.sqrt(np.prod(self.n_blocks))))
		n = int(np.ceil(np.prod(self.n_blocks)/m))

		plt.clf()
		fig, axs = plt.subplots(m+1,n)

		# mnist : 10
		# spikes : 0 (with model 1670333380.4641192)
		sample_index = 8#10#0#0#39#51
		img_in = torch.unsqueeze(x[sample_index,:,:,:],axis=0)
		z = self.encoder(img_in.to(device))

		img_out = self.decoder(z)
		if hasattr(self, 'n_channels_in'):
			img_total = np.zeros((self.img_size[0],self.img_size[1],self.n_channels_in))
		else:
			img_total = np.zeros((self.img_size[0],self.img_size[1],1))

		for k in range(0,np.prod(self.n_blocks)):
			i = int(np.floor(float(k)/float(n)))
			j = k-i*n
			curr_z = z[:,:, k*self.z_size:((k+1)*self.z_size) ]
			img_out = pytorch_to_numpy_image(self.decoder_block(curr_z))
			if (img_out.ndim == 2): # special case, add channel dimension
				img_out = np.expand_dims(img_out,axis=2)
			img_total = img_total+img_out
			# now normalise for viewing purposes
			img_out = (255.*np.clip( img_out,0.0,1.0) ).astype(int)#(255.*img_out/(img_out.max())).astype(int)
			img_out = np.tile(img_out,(1,1,3))
			axs[i,j].imshow( img_out.astype(np.uint8))
			axs[i,j].xaxis.set_ticklabels([])
			axs[i,j].yaxis.set_ticklabels([])
			if (output_dir != ''):
				imageio.imwrite(output_dir+'img_decomposition_block_'+str(k).zfill(3)+'.png',img_out.astype(np.uint8))


		img_total = 255.*np.clip( img_total,0.0,1.0)
		img_total = np.tile(img_total,(1,1,3))

		# so that unused dimensions are not displayed
		for k in range(np.prod(self.n_blocks),m*n):
			i = int(np.floor(float(k)/float(n)))
			j = k-i*n
			axs[i, j].axis('off')

		# now show input and output
		axs[m,0].imshow( pytorch_to_numpy_image(img_in),cmap="gray")
		axs[m,0].xaxis.set_ticklabels([])
		axs[m,0].yaxis.set_ticklabels([])
		img_out = self.forward(img_in.to(device))
		axs[m,1].imshow( img_total.astype(np.uint8))
		#axs[m,1].imshow( pytorch_to_numpy_image(img_out),cmap="gray")
		axs[m,1].xaxis.set_ticklabels([])
		axs[m,1].yaxis.set_ticklabels([])
		#

		for k in range(2,n):
			axs[m, k].axis('off')

		if (output_dir != ''):
			imageio.imwrite(output_dir+'img_decomposition_input.png',pytorch_to_numpy_image(img_in))
			imageio.imwrite(output_dir+'img_decomposition_output.png',pytorch_to_numpy_image(img_out))

		plt.tight_layout()
		if (output_dir != ''):
			plt.savefig(output_dir+'img_decomposition.pdf')
		else:
			plt.show()


	def navigate_latent_codes(self,output_dir=''):
		n_block =4
		n_interpolate = 6
		n_img_example = 100
		# model : 1667503520.1869738


		if(hasattr(self,'data_test_dir')):
			x,_ = get_data_batch(self.data_test_dir,batch_size=256,shuffle_samples=0)
		else:
			x,_ = get_data_batch(data_test_root_dir+self.training_object+'_test/',batch_size=256,shuffle_samples=0)
		z = self.encoder(x.to(device))
		plt.figure()
		plt.imshow(pytorch_to_numpy_image(torch.unsqueeze(x[n_img_example,:,:,:],dim=1)),cmap='gray')


		# get max and min for each latent dimension
		z_0 = torch.unsqueeze(z[n_img_example,:,:].clone(),axis=0)
		z_0[:,:, (n_block*self.z_size):((n_block+1)*self.z_size)] = torch.tensor( np.asarray([-2.697,-0.281,0.00]) )
		z_min = np.asarray([-2.697,-2.957,-0.340])
		z_max = np.asarray([-1.597,0.270,0.614])
		#block_list = np.random.choice(np.arange(0,np.prod(self.n_blocks)),n_blocks_disp)


		fig, axs = plt.subplots(self.z_size,n_interpolate)

		for i in range(0,self.z_size):
			z_interpolate_list = np.linspace(z_min[i],z_max[i],n_interpolate)
			for k in range(0,n_interpolate):
				z_in = z_0.clone()
				z_in[:,:,n_block*self.z_size+i] = z_interpolate_list[k]
			
				img_out = pytorch_to_numpy_image(self.decoder(z_in))
				img_out = (255.*np.clip( img_out,0.0,1.0) ).astype(int)
				img_out = np.tile(np.expand_dims(img_out,axis=2),(1,1,3))

				if (output_dir != ''):
					imageio.imwrite(output_dir+'img_navigation_block_'+str(n_block).zfill(3)+'_code_'+str(i).zfill(3)+'_img_'+str(k).zfill(3)+'.png',img_out.astype(np.uint8))
				else:
					if (img_out.ndim == 2):
						axs[i,k].imshow( img_out.astype(np.uint8),cmap="gray")
					else:
						axs[i,k].imshow( img_out.astype(np.uint8))


		# plt.tight_layout()
		# if (output_dir != ''):
		# 	plt.savefig(output_dir+'latent_code_navigation.pdf')
		# else:
		# 	plt.show()

	def interpolate_latent_codes(self,output_dir=''):
		n_test = 1
		n_step_interpolation = 10
		sigma= 2.8

		if(hasattr(self,'data_test_dir')):
			# for mnist : 80
			# for spikes
			y,_ = get_data_batch(self.data_test_dir,batch_size=2*n_test,shuffle_samples=10)
		else:
			y,_ = get_data_batch(data_test_root_dir+self.training_object+'_test/',batch_size=2*n_test,shuffle_samples=15)
		z = self.encoder(y)


		plt.clf()
		fig, axs = plt.subplots(n_test,n_step_interpolation)
		if (n_test==1):
			axs = np.reshape(axs,(n_test,n_step_interpolation))

		for i in range(0,n_test):
			if 'spikes' in self.training_object:
				spike_centres = np.asarray([
					[self.img_size[0]/4.0,self.img_size[0]/4.0],[3*self.img_size[0]/4.0,3*self.img_size[0]/4.0]])#self.img_size * np.random.random((2,2))

				[x_grid,y_grid] = np.meshgrid(range(0,self.img_size[1]),range(0,self.img_size[0]))
				y_0 = 1/(np.sqrt(2*np.pi) * sigma**4) * np.exp( - ((y_grid-spike_centres[0,0])**2 + (x_grid-spike_centres[0,1])**2)/(2*sigma**2) )
				y_0 = torch.unsqueeze(torch.unsqueeze(torch.tensor(y_0).float(),axis=0),axis=0)
				y_1 = 1/(np.sqrt(2*np.pi) * sigma**4) * np.exp( - ((y_grid-spike_centres[1,0])**2 + (x_grid-spike_centres[1,1])**2)/(2*sigma**2) )
				y_1 = torch.unsqueeze(torch.unsqueeze(torch.tensor(y_1).float(),axis=0),axis=0)
			else:
				y_0 = torch.unsqueeze(y[i*2,:,:,:],axis=0)
				y_1 = torch.unsqueeze(y[i*2+1,:,:,:],axis=0)
			z_0 = self.encoder(y_0)
			z_1 = self.encoder(y_1)

			lambda_interpolation = np.linspace(0.0,1.0,n_step_interpolation)
			for j in range(0,n_step_interpolation):
				z = lambda_interpolation[j]*z_0 + (1-lambda_interpolation[j])*z_1
				img_out = pytorch_to_numpy_image(self.decoder(z))
				if (img_out.ndim == 2):
					axs[i,j].imshow( img_out,cmap="gray")
				else:
					axs[i,k].imshow( img_out)
				if (output_dir != ''):
					imageio.imwrite(output_dir+'latent_interpolation_img_'+str(j).zfill(3)+'.png',img_out)

				#imageio.imwrite('img_out_interpolate_z_ex_'+str(i)+'_step_'+str(j)+'.png',img_out)
		if (output_dir != ''):
			plt.savefig(output_dir+'latent_code_interpolation.pdf')
		else:
			plt.show()

	def estimate_spike_params(self):

		# read test spikes parameters
		spikes_params_file = 'data/spikes_repulsive_all_k_test/spikes_repulsive_all_k_test.pkl'
		with open(spikes_params_file, 'rb') as f:
			theta = pickle.load(f)
		

		if(hasattr(self,'data_test_dir')):
			y,_ = get_data_batch(self.data_test_dir,batch_size=256,shuffle_samples=0)
		else:
			y,_ = get_data_batch(data_test_root_dir+self.training_object+'_test/',batch_size=256,shuffle_samples=0)
		z = self.encoder(y.to(device))

		# calculate covariance between parameters and z
		z = reshape_latent_code(z,self.z_size)
		
		pdb.set_trace()

	def plot_latent_codes(self):
		n_test = 1000
		

		if(hasattr(self,'data_test_dir')):
			y,_ = get_data_batch(self.data_test_dir,batch_size=self.batch_size,shuffle_samples=0)
		else:
			y,_ = get_data_batch(data_test_root_dir+self.training_object+'_test/',batch_size=self.batch_size,shuffle_samples=0)
		z = pytorch_to_numpy(self.encoder(y.to(device)))

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(z[:,0], z[:,1], z[:,2], c='r', marker='o')
		plt.plot
		plt.show()

	def hist_group_sparsity(self):
		n_test = 1000
		

		if(hasattr(self,'data_test_dir')):
			y,_ = get_data_batch(self.data_test_dir,batch_size=self.batch_size,shuffle_samples=0)
		else:
			y,_ = get_data_batch(data_test_root_dir+self.training_object+'_test/',batch_size=self.batch_size,shuffle_samples=0)
		z = self.encoder(y.to(device))

		#
		z_temp = np.reshape(pytorch_to_numpy(z)[8,:,:],(120))

		plt.figure()
		for i in range(0,np.prod(self.n_blocks)):
			plt.plot(np.abs(z_temp[i*self.z_size:(i+1)*self.z_size]))
			print(np.abs(z_temp[i*self.z_size:(i+1)*self.z_size]).sum())
		plt.legend()
		plt.show()


		# z_group_norm = torch.zeros(self.batch_size,np.prod(self.n_blocks))
		# for i in range(0,np.prod(self.z_size)):
		# 	z_group_norm[:,i] = torch.sqrt( (z[:,:,i*self.z_size:((i+1)*self.z_size)]**2).sum() )

		# z_group_norm = pytorch_to_numpy(torch.mean(z_group_norm,axis=0))
		# plt.plot(z_group_norm)
		# plt.show()




	def train(self):

		#X_train = get_full_data(data_train_root_dir+self.training_object+'_train')
		self.loss_list = []
		
		#self.print_shapes(X_train[0,:,:,:])
		print_step = 1000

		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		loss_fn = torch.nn.MSELoss(reduction='sum')
		min_loss = np.inf

		for i in range(0,self.n_epochs):
			# reset the gradients back to zero
			# PyTorch accumulates gradients on subsequent backward passes
			optimizer.zero_grad()
			
			# get data batch
			if(hasattr(self,'data_train_dir')):
				X_train,theta_train = get_data_batch(self.data_train_dir,batch_size=self.batch_size,shuffle_samples=1)
			else:
				X_train,theta_train  = get_data_batch(data_train_root_dir+self.training_object+'_train/',batch_size=self.batch_size,shuffle_samples=1)
			#self.print_shapes(X_train[0:2,:,:,:].to(device))

			# compute reconstructions
			Y_output = self.forward(X_train.to(device))
			
			# plt.imshow(Y_output[10,0,:,:].data.numpy(), cmap = 'gray')
			# plt.show()
			# compute training reconstruction loss
			train_loss_full = self.loss_fn_custom(Y_output.to(device), X_train.to(device))
			train_loss = train_loss_full[0]
			
			# compute accumulated gradients
			train_loss.backward()
			# perform parameter update based on current gradients
			optimizer.step()
			
			loss_list_temp = []
			for j in range(0,len(train_loss_full)):
				loss_list_temp.append(train_loss_full[j].item())
			self.loss_list.append(loss_list_temp)
			
			# add the mini-batch training loss to epoch loss
			loss = train_loss.item()
			# save the best model 
			if (self.toggle_save_model > 0 and loss < min_loss):
				self.save_model(self.param_dir+'/'+self.model_id)
				print("Saving model")
				min_loss = loss

			# display the epoch training loss
			if (i%print_step == 0):
				print("step : {}/{}".format(i + 1, self.n_epochs))
				print("loss = ",loss_list_temp)
				# save an autoencoding example
				if (self.toggle_save_model > 0 ):
					self.plot_autoencoding(self.results_dir+'iter_'+str(i).zfill(7)+'_',input_output=0)

		if(hasattr(self,'data_test_dir')):
			X_test,theta_test = get_data_batch(self.data_test_dir,batch_size=self.batch_size,shuffle_samples=0)
		else:
			X_test,theta_test  = get_data_batch(data_test_root_dir+self.training_object+'_test/',batch_size=self.batch_size,shuffle_samples=0)
			
		Y_output = self.forward(X_test.to(device))
		test_loss = (self.loss_fn_custom(Y_output.to(device), X_test.to(device)).item())/(float(self.batch_size))
		print("final test loss: ", test_loss)
				
	def eval(self):

		# print(self.encoder)
		# print(self.decoder)
		
		# plt.imshow(Y_output[10,0,:,:].data.numpy(), cmap = 'gray')
		# plt.show()
		# compute training reconstruction loss

		#print_class_instance(self)

		print("lambda regularisation :", (self.lambda_z))

		self.plot_loss()
		self.hist_group_sparsity()
		#self.navigate_latent_codes()
		
		self.plot_autoencoding()
		# self.show_decoder()
		
		#self.navigate_latent_codes()
		#self.navigate_latent_codes()

		# # compute reconstructions
		# Y_output = self.forward(X_test.to(device))
		# test_loss = self.loss_fn_custom(Y_output.to(device), X_test.to(device)).item()
		# print("test loss: ", test_loss)

		self.show_block_decoder()
		# self.plot_autoencoding_stability()
		#self.plot_latent_codes()

	def write_results(self):

		print_class_instance(self)

		#self.navigate_latent_codes()

		output_dir = '/home/alasdair/ownCloud/Projets/Yann_traonmilin/deeprep/atomic_autoencoders/images/'+self.training_object+'/'

		# if ('spikes_' in self.training_object):
		# 	self.estimate_spike_params()
		# 	a=1
		#self.navigate_latent_codes(output_dir)
		self.show_block_decoder(output_dir)#
		# self.plot_autoencoding()
		# self.navigate_latent_codes(output_dir)
		#

		# #get_data_batch(dataset_dir,batch_size,img_size,poly_size=3,is_train=1,shuffle_samples=1):
		# X_test,theta_test = get_data_batch(self.training_object,128,self.img_size,self.poly_size,1,0)
		# if (self.architecture=='mlp'):
		# 	z = self.encoder(X_test.view(-1,np.prod(self.img_size)))
		# else:
		# 	z = self.encoder(X_test.to(device))
		# self.show_decoder(z)

		#self.interpolate_latent_codes(output_dir)
		#
		#
