from numpy import inf

class Parameter(object):
    def __init__(self, name, abbr, desc, lims=(-inf, inf), defv=0):
        self.name = name
        self.abbr = abbr
        self.desc = desc
        self.lims = lims
        self.defv = defv
    
    def lnprior(self, x):
        return 0.
