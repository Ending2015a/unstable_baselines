# --- built in ---
import os
import time
import argparse
# --- 3rd party ---
import ray
from ray import tune


def get_experiment(

):
    return tune.Experiment(

    )


class BaseExperiments():
    def __init__(self, name):
        self.args = self.parse_args()
        self.name = name

    def run(self):
        ray.init(address=self.args.address)
        if isinstance(self.args.env_ids, str):
            self.args.env_ids = [self.args.env_ids]
        
        exp_name = self.get_exp_name()
        experiments = []
        for env_id in self.args.env_ids:
            root = os.path.join(self.args.root, env_id)
            experiments.append(tune.Experiment(
                name = exp_name,
                run = self.run_experiment,
                config = {
                    'root': root,
                    'env_id': env_id,
                    **self.search_config()
                },
                resources_per_trial = {
                    'cpu': self.args.cpu,
                    'gpu': self.args.gpu
                },
                num_samples = self.args.trials,
                local_dir = root,
                trial_dirname_creator = self.get_trial_dirname_creator()
            ))

        tune.run_experiments(
            experiments
        )

    def parse_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument('--address', type=str, default='localhost:6379')
        parser.add_argument('--cpu',     type=int, default=4)
        parser.add_argument('--gpu',     type=int, default=1)
        parser.add_argument('--root',    type=str, default='~/dev/unstable_baselines/log/experiments/')
        parser.add_argument('--exp_name',type=str, default=None)
        parser.add_argument('--trials',  type=int, default=1)
        parser.add_argument('--env_ids', nargs='+', default=['BeamRiderNoFrameskip-v4'])
        return parser.parse_args()

    def search_config(self):
        return {}

    @staticmethod
    def run_experiment(config):
        raise NotImplementedError

    def get_exp_name(self):
        exp_names = []
        if self.name is not None:
            exp_names.append(self.name)
        if self.args.exp_name is not None:
            exp_names.append(self.args.exp_name)
        return '-'.join(exp_names)

    def get_trial_dirname_creator(self):
        def _wrap(trial):
            res = ','.join([
                f'{key}={value}'
                for key, value in sorted(trial.evaluated_params.items())
            ])
            if not res:
                res = trial.trial_id
            else:
                expnum = trial.trial_id.split('_')[1]
                if expnum:
                    res = res + '_' + expnum
            return res
        return _wrap
