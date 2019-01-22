from numpy import ndarray
from .filter import Filter, ClearFilter


class Instrument:
    def __init__(self, name, filters, qes=None):
        self.name = name

        assert all([isinstance(f, Filter) for f in filters]), "All filters must be Filter instances."
        self.filters = filters
        self.pb_n = npb = len(filters)
        self.pb_names = [f.name for f in self.filters]

        if qes is not None:
            if isinstance(qes, (tuple, list, ndarray)):
                assert len(filters) == len(qes), "Number of QE profiles differs from the number of passbands."
                assert all([isinstance(qe, Filter) for qe in qes]), "All QE profiles must be Filter instances."
                self.qes = qes
            elif isinstance(qes, Filter):
                self.qes = npb * [qes]
        else:
            self.qes = npb * [ClearFilter('QE')]