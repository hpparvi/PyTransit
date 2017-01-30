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

    def __repr__(self):
        return '{:20s} {:6s} ({:5.2f} <= v <= {:5.2f})'.format(self.name, '['+self.abbr+']', self.lims[0], self.lims[1])
