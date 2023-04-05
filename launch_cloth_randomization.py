import os
import sys

mode = int(sys.argv[1])
if mode == 0:
    seed = 32432
else:
    seed = 12432
n_output = 10000

i = 0
while i <= n_output:
    print("==>", i)
    output_fn = f"output/cloth_project/perturbed_cloth_{mode}_{i}.obj"
    if not os.path.isfile(output_fn):
        os.system(
            f"python /home/zyuwei/Projects/cloth_shape_estimation/cloth_randomization.py -mode {mode} -i {i} -seed {seed} -n_openmp_thread 9"
        )
        if not os.path.isfile(output_fn):
            seed += n_output
        else:
            i += 1
    else:
        i += 1
