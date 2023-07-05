
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



# def shuffle_list(list_in,random_seed):
# 	list_shuffle = list_in.copy()
# 	random.Random(random_seed).shuffle(list_shuffle)
# 	return list_shuffle
