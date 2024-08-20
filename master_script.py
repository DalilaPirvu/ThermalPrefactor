import os,sys

sys.path.append('/home/dpirvu/python_stuff/')
sys.path.append('/home/dpirvu/project/paper_prefactor/bubble_codes/')
print(sys.path)

import numpy as np

parent_dir = "/home/dpirvu/project/paper_prefactor/"
os.chdir(parent_dir)
print(os.getcwd())

siminit = 2000
step = 1
howmany = 1000
run_the_job = True


# Define the base directory
base_dir = "./out/"
sub_dirs = [f"sim013_len80_{n}" for n in range(siminit, siminit + howmany)]
print(sub_dirs)

# Define the constant values for replacement
constants = np.array([(siminit + n * step, siminit + (n + 1) * step) for n in range(len(sub_dirs))])
print(constants)

# Loop through each sub-directory
for idx, sub_dir in enumerate(sub_dirs):
    # Copy the directory recursively
    os.system(f"scp -r ./w {base_dir}{sub_dir}/")

    # Change directory
    os.chdir(f"{base_dir}{sub_dir}/")

    # Replace values in constants.f90 file
    with open("constants.f90", "r") as file:
        data = file.read()

    # Perform the replacement for the constants
    nums = constants[idx]
    print('writing', nums[0], nums[1])
    data = data.replace(f"lSim = {siminit}, nSim = {siminit+step}", f"lSim = {nums[0]}, nSim = {nums[1]}")

    # Write the modified data back to the file
    with open("constants.f90", "w") as file:
        file.write(data)


    # Replace values in batch file
    with open("batch", "r") as file:
        data = file.read()

    # Perform the replacement for the constants
    minsim = nums[0]
    print('job name', minsim)
    data = data.replace(f"#SBATCH -J sim{siminit}", f"#SBATCH -J sim{minsim}")

    # Write the modified data back to the file
    with open("batch", "w") as file:
        file.write(data)
        
    # Submit batch job
    if run_the_job:
        os.system("sbatch batch")

    # Change directory back
    os.chdir(f"{parent_dir}")
