import os
import subprocess

python_cmd = [
    'python',
    'test.py',
    '--input_dir=/input_images',
    '--output_file=/output_data/test_out.csv'
]

cmd = [
    'nvidia-docker', 'run',
    '-v', '{}:/input_images'.format(os.path.abspath('/home/aleksey/code/nips2017-nw/dataset/images/')),
    '-v', '{}:/output_data'.format(os.path.abspath(os.getcwd())),
    '-v', '{}:/code'.format(os.path.abspath(os.getcwd())),
    '-v', '{}:/checkpoints'.format(os.path.abspath('/home/aleksey/code/nips2017-nw/shared/checkpoints/')),
    '-w', '/code',
    'rwightman/pytorch-extra'
]

cmd.extend(python_cmd)

subprocess.call(cmd)