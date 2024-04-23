from common_packages import *

class IgoPreconditionerGeneratorBase(ABC):

    def __init__(self, params):
        pass

    @abstractmethod
    def generate(self, igo, old_cols, new_cols, A_updated_rows, A_new_rows, params):
        raise NotImplementedError
