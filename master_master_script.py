import os,sys

sys.path.append('/home/dpirvu/python_stuff/')
sys.path.append('/home/dpirvu/project/paper_prefactor/bubble_codes/')
print(sys.path)

import numpy as np

siminit = 400
offset = 1000
cores = 40
step = 1
run_the_job = False


# Define the base directory
base_dir = "/home/dpirvu/project/paper_prefactor/out/"
os.chdir(base_dir)
print(os.getcwd())

multiple = [nn for nn in range(2)]

for times in multiple:
    # Define the constant values for replacement
    sub_dirs = [f"simulations{n}" for n in range(siminit + offset + cores * times, \
                                                 siminit + offset + cores * times + cores)]
    print(sub_dirs)

    constants = np.array([(siminit + offset + cores * times + n * step, \
                           siminit + offset + cores * times + (n + 1) * step) for n in range(len(sub_dirs))])
    print(constants.flatten())
    
    newinit = constants[0][0]
    print(newinit)

    # Loop through each sub-directory
    for idx, sub_dir in enumerate(sub_dirs):
        # Copy the directory recursively
        os.system(f"scp -r ../w {base_dir}{sub_dir}/")

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
        os.system(f"make clean")
        os.system(f"make")

        # Change directory back
        os.chdir(f"{base_dir}")

    os.system(f"scp ./batch_file ./batch_file{newinit}")

    # Replace values in batch file
    with open(f"batch_file{newinit}", "r") as file:
        job = file.read()

    job = job.replace(f"-J sim{siminit}", f"-J sim{newinit}")
    job = job.replace(f"out/slurm-{siminit}", f"out/slurm-{newinit}")

    # Loop through each sub-directory
    for idx, sub_dir in enumerate(sub_dirs):
        # Perform the replacement for the constants
        nums = constants[idx][0]
        print('job name', nums)

        job = job.replace(f"./simulations{siminit + idx}/", f"./simulations{nums}/")

    # Write the modified data back to the file
    with open(f"batch_file{newinit}", "w") as file:
        file.write(job)

    # Submit batch job
    if run_the_job:
        os.system(f"sbatch batch_file{newinit}")

    # Change directory back
    os.chdir(f"{base_dir}")
