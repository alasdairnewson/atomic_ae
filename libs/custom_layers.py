


import torch
import torch.nn as nn
import pdb
import numpy as np

def reshape_latent_code(z,latent_packet_size): 
	z = z.view((-1,int(z.shape[2]/latent_packet_size),latent_packet_size))
	# re-permute axes to get correct final shape
	z = torch.transpose(z,2,1)
	return(z)

class View(nn.Module):
	def __init__(self, shape):
		super().__init__()
		self.shape = shape

	def forward(self, input):
		'''
		Reshapes the input according to the shape saved in the view data structure.
		'''
		batch_size = input.size(0)
		shape = (batch_size, *self.shape)
		#out = input.view(shape)
		out = input.reshape(shape)
		return out


class Mosaic(nn.Module):
	def __init__(self, z_size, n_blocks, decoder_block):
		super().__init__()
		self.z_size, self.n_blocks, self.decoder_block = z_size, n_blocks, decoder_block
		

	def forward(self, x):
		return mosaic_to_image(x,self.z_size,self.n_blocks,self.decoder_block)

class Decoder_Sum_Layer(nn.Module):
	def __init__(self, z_size, n_blocks, decoder_block):
		super().__init__()
		self.z_size, self.n_blocks, self.decoder_block = z_size, n_blocks, decoder_block
	def forward(self, z):

		y = self.decoder_block(z[ :, :, 0 : self.z_size ])
		for i in range(1,np.prod(self.n_blocks)):
			y = y+self.decoder_block(z[ :, :, i*self.z_size : (i+1)*self.z_size ])
		return y

class Decoder_Max_Layer(nn.Module):
	def __init__(self, z_size, n_blocks, decoder_block):
		super().__init__()
		self.z_size, self.n_blocks, self.decoder_block = z_size, n_blocks, decoder_block
	def forward(self, z):

		y = self.decoder_block(z[ :, :, 0 : self.z_size ])
		for i in range(1,np.prod(self.n_blocks)):
			y = torch.cat((y, self.decoder_block(z[ :, :, i*self.z_size : (i+1)*self.z_size ])) ,dim=1)
		y,_ = torch.max(y,dim=1,keepdim=True)
		return y

class Decoder_Prod_Layer(nn.Module):
	def __init__(self, z_size, n_blocks, decoder_block):
		super().__init__()
		self.z_size, self.n_blocks, self.decoder_block = z_size, n_blocks, decoder_block
	def forward(self, z):

		y = self.decoder_block(z[ :, :, 0 : self.z_size ])
		for i in range(1,np.prod(self.n_blocks)):
			y = torch.mul(y,self.decoder_block(z[ :, :, i*self.z_size : (i+1)*self.z_size ]))
		return y

class Sort_Latent_Layer(nn.Module):
	def __init__(self, latent_packet_size):
		super().__init__()
		self.latent_packet_size = latent_packet_size
	def forward(self, z):
		'''
		Sort latent code according to lexicographic order
		'''
		
		# change 1 and 2 axes to get the correct ordering
		z = z.view((-1,int(z.shape[2]/self.latent_packet_size),self.latent_packet_size))
		
		z_inds = torch.argsort( (z)[:,:,0],axis=1 )

		z_ordered = z.clone()
		for i in range(0,z.shape[0]):
			z_temp = z[i,:,:]
			z_ordered[i,:,:] = z_temp[z_inds[i]]

		# re-permute axes to get correct final shape
		z_ordered = z_ordered.reshape((z.shape[0],1,z.shape[1]*z.shape[2]))
		return(z_ordered)
