import os
import argparse

from util.args import save_args, load_args

class Log:

    def __init__(self, log_dir: str) -> object:

        self._log_dir = log_dir
        self._logs = dict()

        # make sure the log directories exist
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.isdir(self.metadata_dir):
            os.mkdir(self.metadata_dir)
        if not os.path.isdir(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        open(self.log_dir + '/log.txt', 'w').close()

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def checkpoint_dir(self):
        return self._log_dir + '/checkpoints'

    @property
    def metadata_dir(self):
        return self._log_dir + '/metadata'

    def log_message(self, msg: str):
        with open(self.log_dir + '/log.txt', 'a') as f:
            f.write(msg+"\n")

    def create_log(self, log_name: str, key_name: str, *value_names):
        if log_name in self._logs.keys():
            raise Exception('Log already exists!')
        self._logs[log_name] = (key_name, value_names)
        # write to csv
        with open(self.log_dir + f'/{log_name}.csv', 'w') as f:
            f.write(','.join((key_name,) + value_names) + '\n')

    def log_values(self, log_name, key, *values):
        if log_name not in self._logs.keys():
            raise Exception('Log not existent!')
        if len(values) != len(self._logs[log_name][1]):
            raise Exception('Not all required values are logged!')
        # write to csv
        with open(self.log_dir + f'/{log_name}.csv', 'a') as f:
            f.write(','.join(str(v) for v in (key,) + values) + '\n')

    def log_args(self, args: argparse.Namespace):
        save_args(args, self._log_dir)

    def log_learning_rates(self, optimizer):
        self.log_message("Learning rate net: " + str(optimizer.param_groups[0]['lr']))
        self.log_message("Learning rate net 1x1 conv: " + str(optimizer.param_groups[1]['lr']))


