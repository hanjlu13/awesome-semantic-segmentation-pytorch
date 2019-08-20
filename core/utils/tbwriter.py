from tensorboardX import SummaryWriter


class TensorboardWriter(object):
    """Writes for writing training info to file"""

    @staticmethod
    def init(log_dir):
        """
        Args:
            log_dir (str): path to log directory 
        """
        TensorboardWriter.writer = SummaryWriter(log_dir=log_dir)

    @staticmethod
    def write_scalar(names, values, iter):
        # TODO: refactor in an more efficient way
        if type(names) is not list:
            names = [names]
        if type(values) is not list:
            values = [values]
        for var_name, var_value in zip(names, values):
            TensorboardWriter.writer.add_scalar(var_name, var_value, iter)

    @staticmethod
    def write_scalars(names, values, iter):
        # TODO: refactor in an more efficient way
        if type(names) is not list:
            names = [names]
        if type(values) is not list:
            values = [values]
        for var_name, var_value in zip(names, values):
            TensorboardWriter.writer.add_scalars(var_name, var_value, iter)

    @staticmethod
    def add_histogram(name, param, iter):
        TensorboardWriter.writer.add_histogram(name, param, iter)
