import numpy as np
import csv
import os
import run_attacks_and_defenses
import yaml

with open('../python/local_config.yaml', 'r') as f:
    local_config = yaml.load(f)
main_dir = local_config['results_dir']

attack_dir = os.path.join(main_dir, 'attacks')
targeted_attack_dir = os.path.join(main_dir, 'targeted_attacks')
defense_dir = os.path.join(main_dir, 'defenses')
output_dir = main_dir

def load_defense_output(filename):
  """Loads output of defense from given file."""
  result = {}
  with open(filename) as f:
    for row in csv.reader(f):
      try:
        image_filename = row[0]
        if image_filename.endswith('.png') or image_filename.endswith('.jpg'):
          image_filename = image_filename[:image_filename.rfind('.')]
        label = int(row[1])
      except (IndexError, ValueError):
        continue
      result[image_filename] = label
  return result


def write_score_matrix(filename, scores, mask, row_names, column_names):
    """Helper method which saves score matrix."""
    scores_obj = scores.astype(np.object)
    scores_obj[np.logical_not(mask)] = 'na'
    result = np.pad(scores_obj, ((1, 0), (1, 0)), 'constant').astype(np.object)

    result[0, 0] = ''
    result[1:, 0] = row_names
    result[0, 1:] = column_names
    np.savetxt(filename, result, fmt='%s', delimiter=',')

attack_names = sorted(os.listdir(os.path.join(main_dir, 'attacks')))
attack_names_idx = {name: index for index, name in enumerate(attack_names)}
n_attacks = len(attack_names)
targeted_attack_names = sorted(os.listdir(os.path.join(main_dir, 'targeted_attacks')))
targeted_attack_names_idx = {name: index
                               for index, name
                               in enumerate(targeted_attack_names)}
n_targeted_attacks = len(targeted_attack_names)
defense_names = sorted(os.listdir(os.path.join(main_dir, 'defenses')))
defense_names_idx = {name: index for index, name in enumerate(defense_names)}
n_defenses = len(defense_names)



# In the matrices below: rows - attacks, columns - defenses.
found_defense_outputs_attacks = np.zeros((n_attacks, n_defenses), dtype=np.bool)
found_defense_outputs_targeted_attacks = np.zeros((n_targeted_attacks, n_defenses), dtype=np.bool)
accuracy_on_attacks = np.zeros((n_attacks, len(defense_names)), dtype=np.int32)
accuracy_on_targeted_attacks = np.zeros((n_targeted_attacks, len(defense_names)), dtype=np.int32)
hit_target_class = np.zeros((n_targeted_attacks, len(defense_names)), dtype=np.int32)

dataset_meta = run_attacks_and_defenses.DatasetMetadata('../data/dev_dataset.csv')

for defense_idx, defense_name in enumerate(defense_names):
    defense_output_dir = os.path.join(defense_dir, defense_name)
    defense_output_dir_attacks = os.path.join(defense_output_dir, 'attacks')
    defense_output_dir_targeted_attacks = os.path.join(defense_output_dir, 'targeted_attacks')

    if os.path.exists(defense_output_dir_attacks):
        defense_output_files_attacks = os.listdir(defense_output_dir_attacks)
        for f in defense_output_files_attacks:
            result_dict = load_defense_output(os.path.join(defense_output_dir_attacks, f))
            attack_name = f[:-4] # remove .csv
            if attack_name in attack_names_idx:
                attack_idx = attack_names_idx[attack_name]

                found_defense_outputs_attacks[attack_idx, defense_idx] = True
                for image_id, predicted_label in result_dict.items():
                    true_label = dataset_meta.get_true_label(image_id)
                    if true_label == predicted_label:
                        accuracy_on_attacks[attack_idx, defense_idx] += 1

    if os.path.exists(defense_output_dir_targeted_attacks):
        defense_output_files_targeted_attacks = os.listdir(defense_output_dir_targeted_attacks)
        for f in defense_output_files_targeted_attacks:
            result_dict = load_defense_output(os.path.join(defense_output_dir_targeted_attacks, f))
            attack_name = f[:-4] # remove .csv
            if attack_name in targeted_attack_names_idx:
                attack_idx = targeted_attack_names_idx[attack_name]

                found_defense_outputs_targeted_attacks[attack_idx, defense_idx] = True
                for image_id, predicted_label in result_dict.items():
                    true_label = dataset_meta.get_true_label(image_id)
                    if true_label == predicted_label:
                        accuracy_on_targeted_attacks[attack_idx, defense_idx] += 1
                    target_class = dataset_meta.get_target_class(image_id)
                    if target_class == predicted_label:
                        hit_target_class[attack_idx, defense_idx] += 1

# Save matrices.
write_score_matrix(os.path.join(output_dir, 'accuracy_on_attacks.csv'),
                   accuracy_on_attacks,
                   found_defense_outputs_attacks,
                   attack_names, defense_names)
write_score_matrix(os.path.join(output_dir, 'accuracy_on_targeted_attacks.csv'),
                   accuracy_on_targeted_attacks,
                   found_defense_outputs_targeted_attacks,
                   targeted_attack_names, defense_names)
write_score_matrix(os.path.join(output_dir, 'hit_target_class.csv'),
                   hit_target_class,
                   found_defense_outputs_targeted_attacks,
                   targeted_attack_names, defense_names)


import pandas as pd
df = pd.read_csv(os.path.join(output_dir, 'hit_target_class.csv'), index_col=0)
if not np.all(df.dtypes == np.int64):
    df.replace(to_replace=['na'], value=[9999], inplace=True)
weights = [22.8, 9.3, 2.0]
columns = ['base_inception_v3_tf', 'base_adv_inception_resnet_v2', 'base_adv_inception_v3']
norm_weights = [x / sum(weights) for x in weights]
weighted_score = np.array(sum([w * df[c].astype(np.float) for w, c in zip(norm_weights, columns)]),dtype=np.int)
out_series = pd.DataFrame(index=df.index, data={'weighted':weighted_score, 'total':df.as_matrix().astype(np.int).sum(axis=1)})
out_series.to_csv(os.path.join(output_dir, 'hit_target_class_weighted.csv'))


df = pd.read_csv(os.path.join(output_dir, 'accuracy_on_attacks.csv'), index_col=0)
if not np.all(df.dtypes == np.int64):
    df.replace(to_replace=['na'], value=[9999], inplace=True)
weights = [30.23, 9.2, 13.4]
columns = ['base_inception_v3_tf', 'base_adv_inception_resnet_v2', 'base_adv_inception_v3']
norm_weights = [x / sum(weights) for x in weights]
weighted_score = np.array(sum([w * df[c].astype(np.float) for w, c in zip(norm_weights, columns)]),dtype=np.int)
out_series = pd.DataFrame(index=df.index, data={'weighted':weighted_score, 'total':df.as_matrix().astype(np.int).sum(axis=1)})
out_series.to_csv(os.path.join(output_dir, 'accuracy_on_attacks_weighted.csv'))
