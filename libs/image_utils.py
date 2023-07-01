
import numpy as np
import math
import pdb
import scipy.misc
import scipy
import sys
import glob, os
import pickle
import random
import imageio
from pathlib import Path

from matplotlib import pyplot as MPL
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import rc


def print_class_instance(x):
	attrs = vars(x)
	print(''.join("%s: %s\n" % item for item in attrs.items()))

def read_image(img_file):
	img_out = imageio.imread(img_file)
	img_out = img_out/255.0

	# special case for images of number of colour channels  = 1
	if (img_out.ndim == 2):
		img_out = np.expand_dims(img_out,axis=2)

	return(img_out)

def add_image_border(img_in):
	img_size = img_in.shape
	img_in[0,:] = 1
	img_in[:,0] = 1
	img_in[img_size[0]-1,:] = 1
	img_in[:,img_size[1]-1] = 1
	return img_in

def write_image_encoding(sess,ae,img_in,img_name):
	y,encoding = sess.run([ae['y'],ae['encoding']], feed_dict={ae['x']: img_in})
	n_layers = len(encoding)
	for i_layer in range(0,n_layers-1):
		curr_imgs = encoding[i_layer]
		for j_response in range(0,curr_imgs.shape[3]):
			curr_img = np.squeeze(curr_imgs[0,:,:,j_response])
			curr_img = curr_img-curr_img.min()
			scipy.misc.imsave(img_name+'_encoding_layer_'+str(i_layer)+'_response_'+str(j_response).zfill(3)+
				'.png',scipy.misc.imresize(curr_img,(64,64),'nearest')) #s

def write_image_decoding(decoding,img_name):
	n_layers = len(decoding)

	for i_layer in range(0,n_layers):
		curr_imgs = decoding[i_layer]
		for j_response in range(0,curr_imgs.shape[3]):
			plt.figure()
			curr_img = np.squeeze(curr_imgs[0,:,:,j_response])
			curr_img = curr_img-curr_img.min()
			img_output_name = img_name+'_decoding_layer_'+str(i_layer)+'_response_'+str(j_response).zfill(3)+'.png'
			plt.plot(curr_img)
			plt.savefig(img_output_name)

def write_encoder_weights(encoder,img_name):
	n_layers = len(encoder)
	for i_layer in range(0,n_layers):
		curr_weights = encoder[i_layer]
		write_weight_image(curr_weights,img_name+'_autoencoding_layer_'+str(i_layer))
		# for j_response in range(0,curr_weights.shape[2]):
		#     for k_response in range(0,curr_weights.shape[3]):
		#         curr_weight = np.squeeze(curr_weights[:,:,j_response,k_response])
		#         curr_weight = curr_weight-curr_weight.min()
				# scipy.misc.imsave(img_name+'_encoder_weights_layer_'+str(i_layer)+'_filter_'+str(k_response).zfill(3)+
				#     '_depth_'+str(j_response).zfill(3)+'.png',scipy.misc.imresize(curr_weight,(64,64),'nearest')) #s


def hist_encoder_weights(sess,ae,img_in,img_name):
	encoder = sess.run(ae['encoder'], feed_dict={ae['x']: img_in})
	#plt.hist(a, bins='auto')  # plt.hist passes it's arguments to np.histogram

	encoder_weights = []

	n_layers = len(encoder)
	for i_layer in range(0,n_layers):
		curr_weights = encoder[i_layer]
		encoder_weights.append(curr_weights.flatten())
		# for j_response in range(0,curr_weights.shape[2]):
		#     for k_response in range(0,curr_weights.shape[3]):
		#         curr_weight = np.squeeze(curr_weights[:,:,j_response,k_response])
		#         curr_weight = curr_weight-curr_weight.min()
				# scipy.misc.imsave(img_name+'_encoder_weights_layer_'+str(i_layer)+'_filter_'+str(k_response).zfill(3)+
				#     '_depth_'+str(j_response).zfill(3)+'.png',scipy.misc.imresize(curr_weight,(64,64),'nearest')) #s
	plt.hist(np.hstack(encoder_weights), bins=30)
	plt.show()
	#pdb.set_trace()

def show_weight_image(img_in,img_name=''):
	from mpl_toolkits.mplot3d.axes3d import Axes3D

	output_image_dim = 3
	if (output_image_dim ==2):
		img_filter_resize = 64
		img_space = 2

		img_out = np.zeros(( (img_filter_resize+img_space)*img_in.shape[3],(img_filter_resize+img_space)*img_in.shape[2]))
		i_index = 0
		j_index = 0
		for i_response in range(0,img_in.shape[3]):
			j_index = 0
			for j_response in range(0,img_in.shape[2]):
				curr_img = np.squeeze(img_in[:,:,j_response,i_response])
				curr_img = curr_img-curr_img.min()
				img_out[ i_index:(i_index+img_filter_resize), j_index:(j_index+img_filter_resize)] = \
					scipy.misc.imresize(curr_img,(img_filter_resize,img_filter_resize),'nearest')
				j_index = j_index+img_filter_resize+img_space
			i_index = i_index+img_filter_resize+img_space

		#create colour image
		#pdb.set_trace()
		img_out = np.tile(img_out[:,:,None], [1,1,3])
		maxVal = img_out.max()

		i_index = img_filter_resize
		j_index = 0
		for i_response in range(0,img_in.shape[3]):
			j_index = img_filter_resize
			for j_response in range(0,img_in.shape[2]):
				img_out[ :, j_index:(j_index+img_space),1] = maxVal
				j_index = j_index+img_filter_resize+img_space
			img_out[ i_index:(i_index+img_space), :,1] = maxVal
			i_index = i_index+img_filter_resize+img_space

		#add first border
		img_out_temp = np.zeros((img_out.shape[0]+img_space,img_out.shape[1]+img_space,3))
		img_out_temp[img_space:,img_space:,:] = img_out
		img_out_temp[0:img_space,:,1] = maxVal
		img_out_temp[:,0:img_space,1] = maxVal

		if(img_name==''):
			plt.imshow(img_out_temp)
			plt.show()
		else:
			scipy.misc.imsave(img_name+'_weights.png',img_out_temp)
	else:
		m = img_in.shape[3]
		n = img_in.shape[2]

		filter_size_y = img_in.shape[0]
		filter_size_x = img_in.shape[1]

		x = np.arange(0,filter_size_y)
		y = np.arange(0,filter_size_x)
		xs,ys = np.meshgrid(x,y)
		ax = np.zeros((m*n,1))

		z_lim = 1#img_in.max()
		zs = np.zeros((filter_size_y*filter_size_x,1))

		fig = plt.figure(figsize=plt.figaspect(float(m)/n))

		for i in range(0,m):  #output size (number of filters)
			for j in range(0,n):  #input size (depth of the filters)
				#ax[i*n+j] = fig.add_subplot(i+1,j+1,1, projection='3d')
				ax = fig.add_subplot(m,n,i*n+j+1, projection='3d')
				curr_img = img_in[:,:,j,i]
				#normalise weights
				curr_img = curr_img/(np.abs(img_in).max())
				ax.bar3d(xs.flatten(), ys.flatten(), zs.flatten(), np.squeeze(np.ones((filter_size_y*filter_size_x,1))),\
					np.squeeze(np.ones((filter_size_y*filter_size_x,1))), curr_img.flatten(), color='#00ceaa')
				ax.set_zlim3d(-z_lim, z_lim)
	
		if(img_name==''):
			plt.show()
		else:
			plt.savefig(img_name+'_weights.png')
	plt.close()


def show_weights(model,model_id=''):

	result_dir = 'results/'+model_id+'/'
	# retrieve weights
	weight_list = []
	for layer in model:
		if (hasattr(layer, 'weight')==True):
			weights = layer.weight # list of numpy arrays
			weight_list.append(weights[0].detach().cpu().numpy())

			#insert dimension for visualisation purposes
			curr_weights = np.expand_dims(np.expand_dims( weight_list[-1] , axis=-1) , axis=-1)

			if(model_id==''):
				show_weight_image(curr_weights)
			else:
				show_weight_image(curr_weights,result_dir+model_id+'_layer'+str(len(weight_list)))

def plot_weights(model,model_id=''):

	result_dir = 'results/'+model_id+'/'
	# retrieve weights
	weight_list = []
	for layer in model:
		if (hasattr(layer, 'weight')==True):
			weights = layer.weight.detach().numpy().squeeze()
			plt.plot(weights)
			plt.show()

def PCA(data, dims_rescaled_data=2):
	"""
	returns: data transformed in 2 dims/columns + regenerated original data
	pass in: data as 2D NumPy array
	"""
	from scipy import linalg as LA
	m, n = data.shape
	# mean center the data
	data -= data.mean(axis=0)
	# calculate the covariance matrix
	R = np.cov(data, rowvar=False)
	# calculate eigenvectors & eigenvalues of the covariance matrix
	# use 'eigh' rather than 'eig' since R is symmetric, 
	# the performance gain is substantial
	evals, evecs = LA.eigh(R)
	# sort eigenvalue in decreasing order
	idx = np.argsort(evals)[::-1]
	evecs = evecs[:,idx]
	# sort eigenvectors according to same index
	evals = evals[idx]
	# select the first n eigenvectors (n is desired dimension
	# of rescaled data array, or dims_rescaled_data)
	evecs = evecs[:, :dims_rescaled_data]
	#evecs = evecs[:, 1:]
	# carry out the transformation on the data using eigenvectors
	# and return the re-scaled data, eigenvalues, and eigenvectors
	return np.dot(evecs.T, data.T).T, evals, evecs

def plot_pca(data,img_data,colour_list=0,size_list=1,graph_output_name=""):
	import matplotlib._color_data as mcd
	indigo_colour = mcd.CSS4_COLORS["indigo"].upper()

	if (len(data.shape) == 1):
		data_resc = np.transpose(np.vstack((data,data)))
	else:
		if (data.shape[1] >=4):
			data_resc, evals,evecs = PCA(data,5)
			#pdb.set_trace()
			#data_resc = data[:,0:5]
		else:
			data_resc=data
	if (len(colour_list)==1):
		colour_list = np.ones((data.shape[0],1))
	if (len(size_list)==0):
		size_list = 10*np.ones((data.shape[0],1))

	tolerance =  5
	blowup_factor = 10
	min_size = 10
	if (data_resc.shape[1] == 2):
		clr1 =  '#2026B2'
		axis_font_size = 20
		title_font_size = 19
		fig = plt.figure()
		plt.rc('text', usetex=True)
		plt.rc('font', family='serif')
		ax = fig.add_subplot(111)
		#ax = fig.add_subplot(111, projection='3d')#fig.add_subplot(111)
		#ax1.plot(data_resc[:, 0], data_resc[:, 1], '.', mfc=clr1, mec=clr1)
		#p = ax.scatter3D(data_resc[:, 0], data_resc[:, 1],np.zeros((data_resc.shape[0],1)),c=colour_list, cmap='rainbow',picker=tolerance) #colorsMap='jet'
		#fig.colorbar(p, ax=ax)
		#ax = fig.add_subplot(111, projection='3d')#
		#line, = ax.plot(data_resc[:, 0], data_resc[:, 1])#, c=colour_list, ls="", marker="o")#,#ax1.plot(data_resc[:, 0], data_resc[:, 1], '.', mfc=clr1, mec=clr1)
		#line = ax.plot( data_resc[:, 0], data_resc[:, 1], marker="o",c=colour_list,s=np.maximum(blowup_factor*size_list,min_size))
		line = ax.scatter(data_resc[:, 0], data_resc[:, 1], marker="o",s=35,c=indigo_colour)#, c=colour_list) #,s=blowup_factor*size_list)

		#cbar = fig.colorbar(line, ax=ax)
		#cbar.set_label('colour legend', rotation=270)
		#cbar.ax.set_yticklabels(['disk','','','','','square'])

		#plt.rc('font', family='sans-serif')
		#plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
		#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
		#
		#pdb.set_trace()
		plt.xlabel(r"$z_0$", fontsize=axis_font_size)
		plt.ylabel(r"$z_1$", fontsize=axis_font_size, rotation=0, labelpad=15)
		#plt.title(r"Input disk radius, plotted against $z$", fontsize=title_font_size)
		plt.tight_layout()
		plt.draw()


		# line = ax.scatter(data_resc[:, 0],img_data[:,:,:].sum(axis=2).sum(axis=1), marker="o")
	else:
		fig = plt.figure(1, figsize=(4, 3))
		ax = fig.add_subplot(111, projection='3d')
		#ax = Axes3D(fig, rect=[0, 0, .95, 1])#, elev=elev, azim=azim
		#p = ax.scatter3D(data_resc[:, 0], data_resc[:, 1],data_resc[:, 2],c=data_resc[:, 3],s = np.maximum(blowup_factor*(data_resc[:, 4]-min(data_resc[:, 4])),5), cmap='rainbow',picker=tolerance)
		p = ax.scatter3D(data_resc[:, 0], data_resc[:, 1],data_resc[:, 2],s = 15, cmap='rainbow')
		
		# Turn off tick labels
		ax.set_yticklabels([])
		ax.set_xticklabels([])
		ax.set_zticklabels([])
		#plt.tight_layout()
		#p = ax.scatter3D(data_resc[:, 0], data_resc[:, 1],data_resc[:, 2],c=data_resc[:, 3],s = 8, cmap='rainbow',picker=tolerance)
		#p=ax.scatter3D(data_resc[:, 0],data_resc[:, 1],data_resc[:, 2],c=data_resc[:, 3])#,s=np.maximum(blowup_factor*data_resc[:, 4],5),cmap='rainbow')
		#colorsMap='jet' #np.maximum(blowup_factor*data_resc[:, 4],2)
		#fig.colorbar(p, ax=ax)
		#cbar.ax.set_yticklabels(['circles','small squares','large squares'])
	if (graph_output_name==""):
		# create the annotations box
		# im = OffsetImage(img_data[0,:,:], zoom=5)
		# xybox=(50., 50.)
		# ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
		# 		boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
		# # add it to the axes and make it invisible
		# ax.add_artist(ab)
		# ab.set_visible(False)

		# def hover(event):
		# 	# if the mouse is over the scatter points
		# 	if line.contains(event)[0]:
		# 		# find out the index within the array from the event
		# 		ind = line.contains(event)[1]["ind"]
		# 		# get the figure size
		# 		w,h = fig.get_size_inches()*fig.dpi
		# 		ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
		# 		hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
		# 		# if event occurs in the top or right quadrant of the figure,
		# 		# change the annotation box position relative to mouse.
		# 		ab.xybox = (xybox[0]*ws, xybox[1]*hs)
		# 		# make annotation box visible
		# 		ab.set_visible(True)
		# 		# place it at the position of the hovered scatter point
		# 		ab.xy =(data_resc[ind[0],0], data_resc[ind[0],1])
		# 		# set the image corresponding to that point
		# 		im.set_data(img_data[ind[0],:,:])
		# 	else:
		# 		#if the mouse is not over a scatter point
		# 		ab.set_visible(False)
		# 	fig.canvas.draw_idle()

		# # add callback for mouse moves
		# fig.canvas.mpl_connect('motion_notify_event', hover)           
		plt.show()
	else:
		plt.savefig(graph_output_name)

def plot_interpolation(sess,ae,data,first_image_id,second_image_id,file_list,graph_output_name=""):

	y,z = sess.run([ae['y'],ae['z']], feed_dict={ae['x']: np.asarray(data)})
	z = np.squeeze(z)
	y = np.squeeze(y)

	if (len(z.shape) == 1):
		z_size = 1
		z = np.expand_dims(z,axis=1)
		data_temp = np.hstack((z,z))
	else:
		z_size = z.shape[1]

	img_size = 64
	num_examples = 100
	decimalPlaces = 3
	currInd = 0
	# first z
	z_first = z[first_image_id,:]
	#second z
	z_second = z[second_image_id,:]

	z_list = z_first + np.expand_dims(np.linspace(0,1,num_examples),axis=1)*(z_second-z_first)
	x_list_first = sess.run(ae['y'], feed_dict={ae['z']: np.reshape(z_list,(z_list.shape[0],1,1,z_size))})

	n_iters = 10
	z_list_iter = z_list
	x_list = x_list_first
	for ind_i in range(0,n_iters):
		if (z_size == 1):
			x_list_temp = sess.run(ae['y'], feed_dict={ae['z']: np.reshape(z_list_iter[:,0],(z_list_iter.shape[0],1,1,z_size))})
		else:
			x_list_temp = sess.run(ae['y'], feed_dict={ae['z']: np.reshape(z_list_iter,(z_list_iter.shape[0],1,1,z_size))})
		x_list = np.zeros((x_list_temp.shape[0],img_size*img_size))
		for ind_i in range(0,x_list.shape[0]):
			x_list[ind_i,:] = np.reshape(x_list_temp[ind_i,:,:],(1,img_size*img_size))
		for ind_i in range(0,x_list.shape[0]):
			x_list[ind_i,:] = x_list[ind_i,:]-np.min(x_list[ind_i,:].flatten())
			x_list[ind_i,:] = x_list[ind_i,:]/(np.max(x_list[ind_i,:].flatten()))
		z_list_iter = sess.run(ae['z'], feed_dict={ae['x']: np.asarray(x_list)})

	if (len(z_list_iter.shape) == 1):
		z_list_iter = np.expand_dims(z_list_iter,axis=1)
	else:
		z_list_iter = np.squeeze(z_list_iter)
	# plot result
	fig = plt.figure()
	ax = fig.add_subplot(111)

	#set up colours
	first_colour = 1
	second_colour = 5
	third_colour = 10
	colour_list = np.vstack( (first_colour*np.ones((z.shape[0],1)) , \
		second_colour*np.ones((z_list.shape[0],1)),  third_colour*np.ones((z_list_iter.shape[0],1))))

	plot_data = np.vstack( (z[:,0:z_size],z_list[:,0:z_size],z_list_iter[:,0:z_size]) )
	img_data = np.vstack( (y,np.reshape(x_list_first,(x_list.shape[0],img_size,img_size)),np.reshape(x_list,(x_list.shape[0],img_size,img_size))))
	if (z_size == 1):
		plot_data = np.hstack((plot_data,plot_data))
		z = np.hstack((z,z))

	line = ax.scatter(plot_data[:, 0], plot_data[:, 1], s=10,c=colour_list)#,picker=True)
	if (graph_output_name==""):
		# create the annotations box
		im = OffsetImage(img_data[0,:,:], zoom=5)
		xybox=(50., 50.)
		ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
				boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
		# add it to the axes and make it invisible
		ax.add_artist(ab)
		ab.set_visible(False)

		def on_pick(event):
			# if the mouse is over the scatter points

			# find out the index within the array from the event
			#ind = line.contains(event)[1]["ind"]
			xdata, ydata = line.get_data()

			# get the figure size
			w,h = fig.get_size_inches()*fig.dpi
			ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
			hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
			# if event occurs in the top or right quadrant of the figure,
			# change the annotation box position relative to mouse.
			ab.xybox = (xybox[0]*ws, xybox[1]*hs)
			# make annotation box visible
			ab.set_visible(True)
			# place it at the position of the hovered scatter point
			ab.xy =(plot_data[ind[0],0], plot_data[ind[0],1])
			# set the image corresponding to that point
			im.set_data(img_data[ind[0],:,:])

		# 	fig.canvas.draw_idle()
		def hover(event):
			# if the mouse is over the scatter points
			if line.contains(event)[0]:
				# find out the index within the array from the event
				ind = line.contains(event)[1]["ind"]
				# get the figure size
				w,h = fig.get_size_inches()*fig.dpi
				ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
				hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
				# if event occurs in the top or right quadrant of the figure,
				# change the annotation box position relative to mouse.
				ab.xybox = (xybox[0]*ws, xybox[1]*hs)
				# make annotation box visible
				ab.set_visible(True)
				# place it at the position of the hovered scatter point
				ab.xy =(plot_data[ind[0],0], plot_data[ind[0],1])
				# set the image corresponding to that point
				im.set_data(img_data[ind[0],:,:])
			else:
				#if the mouse is not over a scatter point
				ab.set_visible(False)
			fig.canvas.draw_idle()


		# add callback for mouse moves
		fig.canvas.mpl_connect('motion_notify_event', hover)
		#fig.canvas.mpl_connect('pick_event', on_pick)
		plt.show()
	else:
		plt.savefig(graph_output_name)

def print_event_attributes(event):
	for attr in dir(event): 
			print(attr,'=>', getattr(event, attr))

def effective_radius(imgIn):
	img_size = imgIn.shape
	centreVect = np.squeeze(imgIn[:,int(np.floor(img_size[1]/2))])
	centrePoints = (centreVect>(1/4)).ravel().nonzero()[0]
	if (len(centrePoints) == 0):
		return 0
	else:
		#pdb.set_trace()
		return (centrePoints[len(centrePoints)-1]-centrePoints[0])

def create_directory(directory):
	if (os.path.isdir(directory)==0):
		os.mkdir(directory)

def create_new_param_id():
	import time
	ts = time.time()
	#find if id already exists
	param_name = str(ts)

	return param_name

#go through the sub-directories
def find_model_by_id(model_id,model_dir):
	for full_file_path in glob.glob(model_dir+'**/*', recursive=True):
		file_path_split = os.path.split(full_file_path)
		file_path,file_name = file_path_split[0],file_path_split[1]
		if(os.path.isdir(full_file_path)==0 and file_name==model_id):
			return(full_file_path)

	print("Error, model id not found")
	return ""

def get_latest_by_id(model_dir,root_dir):
	max_file_date = 0
	for x in os.listdir(root_dir+model_dir):
		curr_dir = root_dir+model_dir+x
		if (os.path.isdir(curr_dir)):
			file_list = glob.glob(curr_dir+'/*.ckpt*')
			if (len(file_list)>0):
				file_name_temp = max(file_list, key=os.path.getctime)
				file_date = os.path.getctime(file_name_temp)
				if(file_date>max_file_date):
					max_file_date = file_date
					latest_model_id = x

	return latest_model_id

def get_latest_model(model_dir,model_id,n_leading_zeros):
	max_file_date = 0
	model_dir = model_dir + model_id+"/"

	file_list = sorted(glob.glob(model_dir+'*ckpt.meta'))
	if (len(file_list)>=1):
	# 	epoch_list = []
	# 	# parse file name
	# 	for i in range(0,len(file_list)):
	# 		indexEnd = file_list[i].index('.ckpt.meta')
	# 		indexBegin = indexEnd-n_leading_zeros
	# 		epoch_list.append(int(file_list[i][indexBegin:indexEnd]))
	# 	last_epoch = np.max(np.asarray(epoch_list))
	# 	latest_model = model_dir+model_id+'_'+
	# elif(len(file_list) == 1):
		latest_model = file_list[-1]
		latest_model = latest_model[0:-len(".meta")]
	else:
		print('Error in get_latest_model, id : ', model_id)

	return latest_model

def parse_pnorm_filename(fileName):
	rIndexBegin = fileName.index('_r_')
	rIndexEnd = fileName.index('_p_')
	r = int(fileName[(rIndexBegin+3):(rIndexEnd)])

	pIndexBegin = fileName.index('p_')
	pIndexEnd = fileName.index('.png')
	p = int(fileName[(pIndexBegin+3):(pIndexEnd)])
	return r,p

def parse_file_name(fileName):
	base=os.path.basename(fileName)
	out_name = os.path.splitext(base)[0]

	return out_name

def parse_disk_square_filename(fileName):
	rIndexBegin = fileName.index('_radius')
	rIndexEnd = fileName.index('.png')
	r = int(fileName[(rIndexBegin+8):(rIndexEnd)])

	return r

def parse_disk_radius(fileName):
	rIndexBegin = fileName.index('_radius')
	rIndexEnd = fileName.index('.png')
	r = int(fileName[(rIndexBegin+8):(rIndexEnd)])

	return r

def parse_zoom(fileName):
	rIndexBegin = fileName.index('zoom_')
	rIndexEnd = fileName.index('.png')
	r = int(fileName[(rIndexBegin+5):(rIndexEnd)])

	return r

def show_autoencoder(model,x):

	x_size = len(x)

	delta = 0.25
	t = delta*np.asarray(range(0, x_size))

	min_x = np.min(x)
	max_x = np.max(x)
	mean_x = (max_x+min_x)/2.0

	y = np.squeeze(model.predict(np.reshape(mean_x,(1,1,1))))

	fig, ax = plt.subplots()
	#fig = plt.figure()
	plt.subplots_adjust(left=0.25, bottom=0.25)
		
	#plot result
	curr_plot = ax.plot(t,np.squeeze(y))
	ax.set_ylim(-2,2)
	#plt.axis([0, 1, -10, 10])

	axcolor = 'lightgoldenrodyellow'
	axs = []
	slider = []

	axs = plt.axes([0.25, 0.1 , 0.65, 0.03], facecolor=axcolor)

	slider = Slider(axs, 'x : ', min_x, max_x, valinit=mean_x)
	#axamp.xaxis.set_visible(True)
	#axamp.set_xticks(z) 

	def update(val):
		x_temp = slider.val
		#plot data
		y = np.squeeze(model.predict(np.reshape(x_temp,(1,1,1))))
		curr_plot[0].set_ydata(y)
		# axs.relim()
		# axs.autoscale_view()
		#l.set_data(y)
		fig.canvas.draw()
	slider.on_changed(update)

	resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
	button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

	def reset(event):
		slider.reset()
	button.on_clicked(reset)

	# rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
	# radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)

	# def colorfunc(label):
	#     l.set_color(label)
	#     fig.canvas.draw_idle()
	# radio.on_clicked(colorfunc)

	plt.show()

def create_dirac_dataset(n,batch_size,direction, missing=0):
	continuous = True
	if (direction=='forward'):
		if (missing>0):
			x_min = 2.0/5.0 * float(n-1)
			x_max = 3.0/5.0 * float(n-1)
			train_batch = np.random.rand(batch_size)
			#now map the random numbers to the correct interval
			interval_switch = x_min /(x_min + (n-1.0-x_max))
			min_inds = np.nonzero(train_batch<interval_switch)[0]
			max_inds = np.nonzero(train_batch>=interval_switch)[0]
			train_batch[min_inds] = (x_min) * (train_batch[min_inds] / interval_switch)
			train_batch[max_inds] = x_max + (train_batch[max_inds]-interval_switch) / (1 - interval_switch) *(n-1-x_max)
		else:
			if continuous == True:
				train_batch = np.random.rand(batch_size) * (n-1)
			else:
				train_batch = np.random.randint(n,size=batch_size)
		train_batch_ref = np.zeros((batch_size,n))
		
		#triangle shape
		if (continuous == True): #continuous parameter
			for i in range(0,batch_size):
				x = train_batch[i]

				# gaussian approximation
				sigma = 3.0
				y = np.asarray( range(0,n) )
				train_batch_ref[i,:] = np.exp(-np.power(y - x, 2.) / (2 * np.power(sigma, 2.)))

				# triangular approximation

				# floor_x = np.floor(x)
				# ceil_x = np.ceil(x)
				# if(floor_x == ceil_x):
				# 	train_batch_ref[i,x] = 1 	#special case where we fall exactly on an integer
				# else:
				# 	train_batch_ref[i,int(floor_x)] = 1 - (x - floor_x)
				# 	train_batch_ref[i,int(ceil_x)] = 1 - (ceil_x - x)
		else:
			#train_batch_inds = np.ravel_multi_index( (np.asarray(range(0,batch_size)),train_batch_ref_inds) , (batch_size,img_size))
			train_batch_ref[np.asarray(range(0,batch_size)),train_batch] = 1

		train_batch = np.reshape(train_batch,(batch_size,1)).astype('float')
	elif(direction=='backward'):
		train_batch_ref = np.random.randint(n,size=batch_size)
		train_batch = np.zeros((batch_size,n))
		#train_batch_inds = np.ravel_multi_index( (np.asarray(range(0,batch_size)),train_batch_ref_inds) , (batch_size,img_size))
		train_batch[np.asarray(range(0,batch_size)),train_batch_ref] = 1
		train_batch_ref = np.reshape(train_batch_ref,(batch_size,1)).astype('float')
	else:
		print('Error in create_dirac_dataset, unknown diretion option')
	return train_batch,train_batch_ref

def add_missing_border(img_in,min_radius,max_radius,img_size):
	h_img_size = int(np.floor(img_size/2))
	img_out = img_in
	img_out[np.int(min_radius+h_img_size),:] = 1
	img_out[:,np.int(min_radius +h_img_size)] = 1
	img_out[np.int(max_radius +h_img_size),:] = 1
	img_out[:,np.int(max_radius +h_img_size)] = 1

	return img_out

def extract_shape_class(file_name):
	rIndexBegin = file_name.index('img_')+len('img_')
	rIndexEnd = file_name.index('_number')
	shape_class = file_name[rIndexBegin:rIndexEnd]
	return shape_class
def shuffle_list(list_in,random_seed):
	list_shuffle = list_in.copy()
	random.Random(random_seed).shuffle(list_shuffle)
	return list_shuffle


def show_dirac_decoder(sess,ae,img_size):

	# z = [nExamples 1 1 zSize]
	min_x = 0
	max_x = img_size-1
	mean_x = (float(max_x-min_x))/2.0

	#fig, ax = plt.subplots()
	fig = plt.figure()
	#plt.subplots_adjust(left=0.25, bottom=0.25)
	t = np.arange(0.0, 1.0, 0.001)

	y = sess.run(ae['y'], feed_dict={ae['x']: np.reshape(np.asarray(mean_x),(1,1))})

	#plot result
	#l, = plt.plot(x_axis, y.flatten(), lw=2, color='red')
	l, = plt.plot(np.squeeze(y))
	plt.axis([min_x, max_x, -0.2, 1.0])

	axcolor = 'lightgoldenrodyellow'
	slider_list = []
	slider_step = 0.05
	curr_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
	curr_slider = Slider(curr_ax, 'x', min_x, max_x, valinit=mean_x)
	#axamp.xaxis.set_visible(True)
	#axamp.set_xticks(z) 

	def update(val):
		#plot data
		y = sess.run(ae['y'], feed_dict={ae['x']: np.reshape(np.asarray(val),(1,1))})
		l.set_ydata(np.squeeze(y))
		#l.set_data(np.squeeze(y))
		fig.canvas.draw_idle()
	curr_slider.on_changed(update)

	resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
	button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

	def reset(event):
		curr_slider.reset()
	button.on_clicked(reset)

	# rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
	# radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)

	# def colorfunc(label):
	#     l.set_color(label)
	#     fig.canvas.draw_idle()
	# radio.on_clicked(colorfunc)

	plt.show()

def get_parameters(param_file_name):

	parameter_struct = {}
	with open(param_file_name, 'rb') as handle:
		parameter_dict = pickle.load(handle)

		for key in parameter_dict.keys():
			parameter_struct[key] = parameter_dict.get(key)

		# if ('x_size' in parameter_dict):
		# 	parameter_struct['x_size'] = parameter_dict.get('x_size')
		# else:
		# 	parameter_struct['x_size'] = 64
		# if('n_filters' in parameter_dict):
		# 	parameter_struct['n_filters'] = parameter_dict.get('n_filters')
		# else:
		# 	parameter_struct['n_filters'] = 32
		# if ('n_epochs' in parameter_dict):
		# 	parameter_struct['n_epochs'] = parameter_dict.get('n_epochs')
		# else:
		# 	parameter_struct['n_epochs'] = 10000
		# if('batch_size' in parameter_dict):
		# 	parameter_struct['batch_size'] = parameter_dict.get('batch_size')
		# else:
		# 	parameter_struct['batch_size'] = 64
		# if ('learning_rate' in parameter_dict):
		# 	parameter_struct['learning_rate'] = parameter_dict.get('learning_rate')
		# else:
		# 	parameter_struct['object_name'] = 'dirac'
		# if ('learning_rate' in parameter_dict):
		# 	parameter_struct['learning_rate'] = parameter_dict.get('learning_rate')
		# else:
		# 	parameter_struct['learning_rate'] = 0.001

		return parameter_struct