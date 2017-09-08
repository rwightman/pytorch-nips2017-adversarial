import yaml
import pandas as pd
import numpy as np
import os
import shutil

with open('../python/local_config.yaml', 'r') as f:
    local_config = yaml.load(f)
images_dir = local_config['images_dir']

dev_dataset = pd.read_csv(os.path.join(images_dir,'target_class.csv'), header=None)

# Random without replacement
np.random.seed(0)
random_100 = np.random.permutation(np.arange(0,1000,1))[0:100]

dev_dataset_random_100 = dev_dataset.iloc[random_100,:]

subset_dir = os.path.join(images_dir,'..','100')
if not os.path.exists(subset_dir):
    os.makedirs(subset_dir)

dev_dataset_random_100.to_csv(os.path.join(subset_dir, 'target_class.csv'),header=False,index=False)

dev_dataset_random_100.columns

for _, row in dev_dataset_random_100.iterrows():
    im = row[0]
    shutil.copy(
        os.path.join(images_dir, im),
        subset_dir
    )