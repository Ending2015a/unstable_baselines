import os
import json
import argparse

from pathlib import Path

import shutil


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Print Score')
  parser.add_argument('--path', type=str, default='log/', help='Root dir')
  parser.add_argument('--file', type=str, default='*.json', help='filename')
  parser.add_argument('--reverse', action='store_true', help='Reverse sorting')

  a = parser.parse_args()

  json_files = sorted([x for x in Path(a.path).rglob(a.file)])
  j_pairs = []

  for json_file in json_files:
    try:
      with open(str(json_file), 'r') as f:
        j = json.load(f)
        if 'episode_info' in j and 'episode_rewards' in j['episode_info']:
          j_pairs.append( (json_file, j) )
    except:
      print('skip: {}'.format(json_file))

  j_pairs = sorted(j_pairs, key=lambda x:x[1]['episode_info']['episode_rewards'], reverse=a.reverse)

  json_tuples = []

  for j_pair in j_pairs:
    json_file = j_pair[0]
    j = j_pair[1]
    
    rews = j['episode_info']['episode_rewards']
    steps = j['episode_info']['episode_length']
    print('{:8.1f}, {:6d}, {}'.format(rews, steps, json_file))

    json_tuples.append((rews, steps, str(json_file)))

  rews, steps, json_files = zip(*json_tuples)
