


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


class Decoder_Sum_Layer(nn.Module):
	def __init__(self, z_size, n_blocks, decoder_block):
		super().__init__()
		self.z_size, self.n_blocks, self.decoder_block = z_size, n_blocks, decoder_block
	def forward(self, z):

		y = self.decoder_block(z[ :, :, 0 : self.z_size ])
		for i in range(1,np.prod(self.n_blocks)):
			y = y+self.decoder_block(z[ :, :, i*self.z_size : (i+1)*self.z_size ])
		return y