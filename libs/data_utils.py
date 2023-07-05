


import sys
import os
import pickle
import glob

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
        latest_model = file_list[-1]
        latest_model = latest_model[0:-len(".meta")]
    else:
        print('Error in get_latest_model, id : ', model_id)

    return latest_model

def get_parameters(param_file_name):

    parameter_struct = {}
    with open(param_file_name, 'rb') as handle:
        parameter_dict = pickle.load(handle)

        for key in parameter_dict.keys():
            parameter_struct[key] = parameter_dict.get(key)

        return parameter_struct