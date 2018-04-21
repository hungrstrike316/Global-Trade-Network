

def print_single_script_file(inpt,fileID):
	fileID.write('#!/bin/bash\n')
	fileID.write('# Job name:')
	fileID.write('#SBATCH --job-name=net\n')
	fileID.write('#\n')
	fileID.write('# Partition:\n')
	fileID.write('#SBATCH --partition=cortex\n')
	fileID.write('#\n')
	#fileID.write('# Constrain Nodes:\n')
	#fileID.write('#SBATCH --constraint=cortex_nogpu\n')
	fileID.write('#\n')
	fileID.write('# Processors:\n')
	fileID.write('#SBATCH --ntasks=4\n')
	fileID.write('#\n')
	fileID.write('# Memory:\n')
	fileID.write('#SBATCH --mem-per-cpu=3500M\n')
	fileID.write('#\n')
	fileID.write('# Wall clock limit:\n')
	fileID.write('#SBATCH --time=48:0:00\n')
	fileID.write('#\n')
	fileID.write('#\n')
	# fileID.write('#SBATCH -o net.out\n')
	# fileID.write('#\n')
	# fileID.write('#SBATCH -e net.err\n')
	# fileID.write('\n')
	fileID.write('cd ../code\n')
	fileID.write('\n')
	#fileID.write('module load python/anaconda3\n')
	fileID.write('module load python/3.5\n')
	fileID.write('\n')
	fileID.write( str('python Construct_WTnet_Adjacency.py ' + str(inpt) + '\n') )
	fileID.write('\n')
	fileID.write('exit\n')
	fileID.write('EOF\n')



years = range(2008,2010) #range(1962,2015) # = total
fidW = open('../scripts4cluster/run_all_Construct_WTnet_Adjacency', 'w' ) 						      # wrapper script file
fidW.write( str('# Note: Should be sitting in scripts4cluster directory. \n') )
fidW.write( str('# Note: Should run chmod 777 first before runnning. \n') )
fidW.write( str('\n') )

for y in range(0,len(years)):
	
	fidS = open( str( '../scripts4cluster/script_py_Construct_WTnet_Adjacency_' + str(years[y]) ), 'w' ) # single script file
	
	print_single_script_file(years[y],fidS) # 

	fidW.write( str('sbatch script_py_Construct_WTnet_Adjacency_' + str(years[y]) + '\n') )

	# close text file.
	fidS.close()



fidW.close() # close wrapper script file.	
