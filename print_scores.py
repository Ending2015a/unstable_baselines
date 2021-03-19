import os
import json
import argparse

from pathlib import Path


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Print Score')
  parser.add_argument('--path', type=str, default='log/', help='Root dir')
  parser.add_argument('--file', type=str, default='*.json', help='filename')
  parser.add_argument('--reverse', action='store_true', help='Reverse sorting')

  a = parser.parse_args()

  json_files = sorted([x for x in Path(a.path).rglob(a.file)])
  j_pairs = []

  for json_file in json_files:
    with open(str(json_file), 'r') as f:
      j = json.load(f)
      if 'monitor' in j and 'episode_rewards' in j['monitor']:
        j_pairs.append( (json_file, j) )
      

  j_pairs = sorted(j_pairs, key=lambda x:x[1]['monitor']['episode_rewards'], reverse=a.reverse)

  for j_pair in j_pairs:
    json_file = j_pair[0]
    j = j_pair[1]
    
    rews = j['monitor']['episode_rewards']
    steps = j['monitor']['episode_length']
    print('{:8.1f}, {:6d}, {}'.format(rews, steps, json_file))
