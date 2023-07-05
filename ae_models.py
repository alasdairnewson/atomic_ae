
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

# different directories necessary for the code
model_root_dir = 'models/'
data_train_root_dir = 'data/'
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



class autoencoder(nn.Module):
	def __init__(self, model_id='', toggle_save_model = 1, training_object = 'spikes',data_train_dir=''):
		super(autoencoder, self).__init__()

		# parameters
		self.root_dir = ""
		self.training_object = training_object
		self.img_size = (128,128)#(256,256)
		self.block_size = (32,32)
		if (self.training_object == 'spikes_repulsive_all_k'):
			self.n_blocks = (10,1)
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
		self.use_bias = True
		self.alpha = 0.2
		self.batch_size = 64
		self.n_epochs = 1000000
		self.toggle_save_model = toggle_save_model
		if(self.training_object == 'spikes_repulsive_all_k'):
			self.learning_rate = 0.0001#
		else:
			self.learning_rate = 0.0005
		# save the losses
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
		# now deal with the results root directory, which potentially might not exist
		if (os.path.exists(self.root_dir+self.results_root_dir) == 0):
			#create the results directory if necessary
			create_directory(self.root_dir+self.results_root_dir)
		# now deal with the results directory of this training object, which potentially might not exist
		if (os.path.exists(self.root_dir+self.results_root_dir+self.training_object+'/') == 0):
			#create the results directory if necessary
			create_directory(self.root_dir+self.results_root_dir+self.training_object+'/')

		self.results_dir = self.root_dir+self.results_root_dir+self.training_object+'/'+self.model_id+'/'
		if (os.path.exists(self.results_dir) == 0):
			#create the results directory if necessary
			create_directory(self.results_dir)

		# create the model itself
		x_size = np.prod(self.img_size)

		# atomic AE model architecture :
		# CONV -> MLP -> LATENT -> MLP -> CONV
			
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

		# DECODER
		# now create single decoder, which we will apply to each sub-block of the latent code
		self.z_conv_size = self.z_spatial_size*self.z_spatial_size*self.z_channel_size_decoder
		# project from space of size z_conv_size*z_conv_size*z_channel_size to img_size
		n_layer_conv = int(np.log2(np.minimum(self.img_size[0],self.img_size[1])/self.z_spatial_size))
			
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
			deconv_modules.append(nn.LeakyReLU(negative_slope=self.alpha))
			deconv_modules.append(nn.Conv2d(self.channel_list[i-1], self.channel_list[i-1], self.kernel_size, stride=1, padding=1))
			deconv_modules.append(nn.LeakyReLU(negative_slope=self.alpha))
		self.conv_decoder = nn.Sequential(*deconv_modules)

		self.decoder_block =  nn.Sequential(
				self.mlp_decoder,
				View((self.z_channel_size_decoder,self.z_spatial_size,self.z_spatial_size)),
				self.conv_decoder
			)
		# finally 
		self.decoder =  nn.Sequential(
			Decoder_Sum_Layer(self.z_size, self.n_blocks, self.decoder_block)
		)

	def forward(self, x):

		z_full = self.encoder(x)
		# create mosaic list
		x = Decoder_Sum_Layer(self.z_size,self.n_blocks, self.decoder_block)(z_full)
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
		torch.save(self, model_file)


	def loss_fn_custom(self,y, x):

		z = self.encoder(x.to(device))
		mse_loss = torch.nn.MSELoss(reduction='mean')(y,x)
		loss = mse_loss

		return loss

	def plot_autoencoding(self,output_dir='',input_output=0):
		n_test = 5
		n_batch = 3

		m = int(np.round(np.sqrt(n_test)))
		n = int(np.ceil(float(n_test)/float(m)))

		if(hasattr(self,'data_test_dir')):
			# mnist : 10
			y,_ = get_data_batch(self.data_test_dir,batch_size=self.batch_size,shuffle_samples=5)
		else:
			y,_ = get_data_batch(data_test_root_dir+self.training_object+'_test/',batch_size=self.batch_size,shuffle_samples=1)

		fig, axs = plt.subplots(2,n_test)
		for i in range(0,n_test):
			y_in = torch.unsqueeze(y[i+n_batch*n_test,:,:,:],axis=0)
			y_out = pytorch_to_numpy_image( self.forward(y_in.to(device)) )
			# normalise images
			y_in = pytorch_to_numpy_image(y_in)
			y_out = np.clip( y_out,0.0,1.0) 
			
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


	def show_decoder(self,output_dir=''):


		if(hasattr(self,'data_train_dir')):
			x,_ = get_data_batch(self.data_test_dir,batch_size=128,shuffle_samples=0)
		else:
			x,_  = get_data_batch(data_test_root_dir+self.training_object+'_test/',batch_size=128,shuffle_samples=0)

		z = self.encoder(x.to(device))

		delta = 0.25
		self.z_block_display = 4
		sample_index = 100
		img_in = torch.unsqueeze(x[sample_index,:,:,:],axis=0)
		z_0 = self.encoder(img_in.to(device))

		min_z = torch.min(z[:,:,self.z_size*self.z_block_display:(self.z_size*(self.z_block_display+1))],axis=0)[0]
		max_z = torch.max(z[:,:,self.z_size*self.z_block_display:(self.z_size*(self.z_block_display+1))],axis=0)[0]

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
		sample_index = 10
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


	def train(self):

		#X_train = get_full_data(data_train_root_dir+self.training_object+'_train')
		self.loss_list = []
		
		#self.print_shapes(X_train[0,:,:,:])
		print_step = 1000

		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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

			# compute reconstructions
			Y_output = self.forward(X_train.to(device))
			
			# compute training reconstruction loss
			train_loss = self.loss_fn_custom(Y_output.to(device), X_train.to(device))

			# compute accumulated gradients
			train_loss.backward()
			# perform parameter update based on current gradients
			optimizer.step()

			self.loss_list.append(train_loss)
			
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
				print("loss = ",loss)
				# save an autoencoding example
				self.plot_autoencoding(self.results_dir+'iter_'+str(i).zfill(7)+'_',input_output=0)

		if(hasattr(self,'data_test_dir')):
			X_test,theta_test = get_data_batch(self.data_test_dir,batch_size=self.batch_size,shuffle_samples=0)
		else:
			X_test,theta_test  = get_data_batch(data_test_root_dir+self.training_object+'_test/',batch_size=self.batch_size,shuffle_samples=0)
			
		Y_output = self.forward(X_test.to(device))
		test_loss = (self.loss_fn_custom(Y_output.to(device), X_test.to(device)).item())/(float(self.batch_size))
		print("final test loss: ", test_loss)
				
	def eval(self):

		print_class_instance(self)
		
		self.plot_autoencoding()
		self.show_block_decoder()
