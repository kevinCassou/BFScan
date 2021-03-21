#######################################################################
############         Performance scan launcher           ##############
#######################################################################

import os,sys,re
from sklearn.model_selection import ParameterGrid
import pandas as pd
import numpy as np


#### function offset taking into account self focusing shift difference when changing ne1,L1

def xoffset(ne1,L1):
    """
    polynomial law estimated from a scan of self focusing as a fonction of ne1, and xoff to have a_0>1.8 at the end of region 1
    ne1 :cm^-3
    l1  :m
    """
    polx = [-2.99025772e-66,  2.43473830e-44,  1.30273333e-22, -3.77631065e-01]
    xoff = np.poly1d(polx)
    return xoff(ne1*L1)


### 0 - Define path and executables

# The input namelist - Remember, this namelist must NOT contain the parameters clrw and tasks_on_projection in Main block
name_input_namelist = "LWFA_ii-env-bfs-2.py"
name_submission_script = "runSmilei_ii-env-bfs.sh"
#name_input_namelist = "tst3d_laser_wakefield.py"

# necessary paths - they must be absolute paths
starting_directory = os.getcwd()
path_input_namelist = "/ccc/scratch/cont003/smilei/cassouke/BF-TEST-CN2-10/"+name_input_namelist
path_input_submission_script = "/ccc/scratch/cont003/smilei/cassouke/BF-TEST-CN2-5/"+name_submission_script

path_executable_develop = "/ccc/scratch/cont003/smilei/cassouke/BF-TEST-CN2-5/smilei"
path_executable_test_develop = "/ccc/scratch/cont003/smilei/cassouke/BF-TEST-CN2-5/smilei_test"

# save content of input namelist
with open(path_input_namelist, "r") as file:
	namelist_file_content = file.readlines()
# save content of submission_script
with open(path_input_submission_script, "r") as file:
	submission_script_file_content = file.readlines()

### 1 - Prepare dataframe with scan parameters

n_e_1 = 1.0e24*np.array([3.0,4.0,5.0,6.0])
r = np.array([0.33, 0.5, 0.66])
l_1 =  1e-3*np.array([0.3, 0.6, 0.9, 1.2])
x_foc = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
c_N2 =  1e-2*np.array([5])

param_grid = {'n_e_1':n_e_1, 'r':r, 'l_1':l_1,'x_foc':x_foc,'c_N2':c_N2 }

df = pd.DataFrame((ParameterGrid(param_grid)))

#add column Config
df['Config'] = df.index
#rearrange columns cols = list(df.columns.values)
df = df[['Config','n_e_1', 'r', 'l_1', 'x_foc', 'c_N2' ]]

df['x_foc'] =  df['x_foc'].values + xoffset(df['n_e_1'].values,df['l_1'].values)

### 2 - create folders and submit job one by one

# create the folder tree
#directories_to_create = [starting_directory+"/Config_" +str(x) for x in df['Config']]

# create directory tree with necessary files and launch each simulation
#for directory in directories_to_create:


for index, row in df.iterrows():
    directory = starting_directory+"/Config_" +str(int(row['Config']))
	# create the directory and enter it
    print(directory)
    os.makedirs(directory)
    os.chdir(directory)
    # add link to executable smilei and smilei_test
    os.system("ln -s "+path_executable_develop)
    os.system("ln -s "+path_executable_test_develop)
    # prepare namelist
    write_to_namelist =  "config_external = " +str(row.to_dict())
    print("dictionary for namelist",write_to_namelist)
	# !! insert in the original namelist file dictionary with parameters after the line External_config !! 
    with open(name_input_namelist, 'w') as namelist:
        for line in namelist_file_content:
            newline = line
            if "External_config" in line:
                #print(line)
                newline = newline + "\n" + write_to_namelist + "\n"
            namelist.write(newline)
    # set the MPI and threads in the submission script
    with open("submission_script.sh", 'w') as submission_script_file:
        for line in submission_script_file_content:
            submission_script_file.write(line)
    
	# launch the simulation
    os.system("ccc_msub submission_script.sh")

	# go back to the original folder
    os.chdir(starting_directory)
	
			
		

