from common_packages import *

class IgoRowSelectorBase(ABC):
    """
    Selects a set of rows based on some criteria. The selected rows will be used to 
    update the preconditioner
    """

    def __init__(self, params):
        self.params = params

    @abstractmethod
    def select_rows(self, params, **kwargs):
        raise NotImplementedError("select_rows not implemented")
