

from ae_models import *
import pandas as pd

import numpy as np
import pdb
import pickle
import argparse

from shutil import get_terminal_size
pd.set_option('display.width', get_terminal_size()[0])

model_root_dir = 'models/'

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%
if __name__ == '__main__':

	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-tr", "--is_training", type=int, default=1,required=False,
		help="Training or testing [Training]")
	ap.add_argument("-sm", "--save_model", type=int, default=1,required=False,
		help="Whether to save model or not")
	ap.add_argument("-m_id", "--model_id", type=str, required=False, default="",
		help="ID for identifying the experiment")
	ap.add_argument("-tr_ob", "--training_object", type=str, required=False, default="contours",
		help="Type of signal")
	ap.add_argument("-data_train_dir", "--data_train_dir", type=str, required=False, default="",
		help="Root directory of the training data")
	args = vars(ap.parse_args())

	is_training = args["is_training"]
	save_model = args["save_model"]
	model_id = args["model_id"]
	training_object = args["training_object"]
	data_train_dir = args["data_train_dir"]


	if (model_id != ''):
		model_dir = find_model_by_id(model_id, model_root_dir)
		ae_model = torch.load(model_dir,map_location=device)
		print(ae_model)
	if (is_training==1):
		if (model_id == ''):
			ae_model = autoencoder(model_id,save_model,training_object,data_train_dir).to(device)
		print(ae_model)
		ae_model.train()
	elif(is_training==0):
		#ae_model
		ae_model.eval()
