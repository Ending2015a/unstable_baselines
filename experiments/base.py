# --- built in ---
import os
import time
import argparse
# --- 3rd party ---
import ray
from ray import tune

class BaseExperiments():
    def __init__(self, name=None):
        self.args = self.parse_args()
        self.args.root = os.path.join(self.args.root, self.args.env_id)
        self.name = name

    def run(self):
        ray.init(address=self.args.address)
        tune.run(
            self.run_experiment,
            name=self.name,
            resources_per_trial={
                'cpu': self.args.cpu,
                'gpu': self.args.gpu
            },
            config={
                'root': self.args.root,
                'env_id': self.args.env_id,
                'rank': self.args.rank,
                **self.search_config()
            },
            local_dir=self.args.root,
            trial_dirname_creator=self.get_trial_dirname_creator()
        )

    def parse_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument('--address', type=str, default='localhost:6379')
        parser.add_argument('--cpu',     type=int, default=4)
        parser.add_argument('--gpu',     type=int, default=1)
        parser.add_argument('--root',    type=str, default='~/dev/unstable_baselines/log/experiments/')
        parser.add_argument('--env_id',  type=str, default='BeamRiderNoFrameskip-v4')
        parser.add_argument('--rank',    type=int, default=0)
        return parser.parse_args()

    def search_config(self):
        return {}

    @staticmethod
    def run_experiment(config):
        raise NotImplementedError

    def get_trial_dirname_creator(self):
        def _wrap(trial):
            return ','.join([
                f'{key}={value}'
                for key, value in sorted(trial.evaluated_params.items())
            ])

        return _wrap