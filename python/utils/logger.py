import yaml
from copy import deepcopy

# Global variable
quiet = False

class NoLogger:
    def __init__(self):
        # Save the initial value of quiet when the Logger is created
        self.old_quiet_value = quiet

    def __enter__(self):
        # Inside the context manager, set quiet to True
        global quiet
        quiet = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # On exiting the context manager, restore the old value of quiet
        global quiet
        quiet = self.old_quiet_value

# Global function
def log(*args):
    # If quiet is False, print the arguments; otherwise, do nothing
    if not quiet:
        print(*args)


class Logger:
    def __init__(self, fout=None):
        self.data = {}
        self.fout = fout
        self.cur_step = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.write()

    def set_step(self, step):
        self.cur_step = step
        if self.cur_step not in self.data:
            self.data[self.cur_step] = {}

    def unset_step(self):
        self.cur_step = None

    def log(self, d, key=None):
        if key is not None:
            self.data[key] = deepcopy(d)
        if self.cur_step is not None:
            for k, v in d.items():
                self.data[self.cur_step][k] = v

    def write(self):
        if self.fout is not None:
            self.fout.write(yaml.dump(self.data, default_flow_style=True))

class SetUpStepLogger:
    def __init__(self, logger, step):
        self.logger = logger
        self.logger.set_step(step)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.unset_step()

