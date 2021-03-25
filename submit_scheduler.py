# simple scheduler to do no t depass limitation of jobs at IRENE


import os,sys,re
import pandas as pd
import time
import subprocess



def get_running_and_waiting_jobs():

	batcmd="ccc_mpp  -rspnH -u cassouke"
	result = subprocess.check_output(batcmd, shell=True)
	n_running = result.count('\n')
	print("all jobs in queue = ",result,"total numbver of submitted jobs = ",n_running)

	return n_running

starting_directory = '/ccc/scratch/cont003/smilei/cassouke/JOBFLOW-TEST'
path_executable_develop = "/ccc/scratch/cont003/smilei/cassouke/JOBFLOW-TEST/smilei"
path_executable_test_develop = "/ccc/scratch/cont003/smilei/cassouke/JOBFLOW-TEST/smilei_test"

df = pd.read_pickle("df_init.pickle")


max_jobs = 295
N_total = len(df)
N_done  = 0
N_to_submit = 0
jobs_left = N_total



while jobs_left>0:

	N_running_now = get_running_and_waiting_jobs()
	N_to_submit = max_jobs-N_running_now

	#try:
	if N_running_now < max_jobs:
		# if number of jobs_left is small wrt to number of possible jobs. This is the case close to end of the submission chain.
		if jobs_left < N_to_submit:
			N_to_submit = jobs_left
		# submit N_to_submit jobs individually. Same as in initial script 
		for index, row in df.head(N_to_submit).itertuples():
			directory = starting_directory+"/Config_" +str(int(row['Config']))
			# create the directory and enter it
			print(directory)
			os.chdir(directory)
			# add link to executable smilei and smilei_test
			os.system("ln -s "+path_executable_develop)
			os.system("ln -s "+path_executable_test_develop)
			
			# launch the simulation
			os.system("ccc_msub submission_script.sh")
			
			# go back to the original folder
			os.chdir(starting_directory)
			
			# remove submitted row from dataframe
			df.drop(index, inplace=True)

	#	else:
	#		pass
	#except: 
	print("N_running_now",N_running_now,'jobs_left ',jobs_left,'N_to_submit',N_to_submit)
	pass

	jobs_left = len(df)
	# wait and verify again. if possible - submit
	time.sleep(1*60)












