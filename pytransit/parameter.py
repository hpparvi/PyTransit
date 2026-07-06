from numpy import inf, where, atleast_2d, zeros, squeeze, array, stack
from scipy.stats import norm, uniform


class Parameter:
    def __init__(self, name, description='', unit='', prior=None, bounds=(-inf, inf), **kwargs):
        self.name: str = name
        self.description: str = description
        self.unit = unit
        self.prior = prior
        self._bounds: tuple[float, float] = bounds
        self.pid: int = -1

    def __str__(self):
        return f"{self.pid:3d} | {self.name:14s} {str(self.prior):40} [{self.bounds[0]:8.2f} .. {self.bounds[1]:8.2f}]"

    def __repr__(self):
        return str(self)

    @property
    def bounds(self):
        return self._bounds

    def log_prior(self, v):
        return self.prior.logpdf(v)

    def rvs(self, size):
        return self.prior.rvs(size)


class ParameterBlock(list):
    def __init__(self, name, *args):
        super().__init__(*args)
        self.name = name

    def add_parameter(self, name, description='', unit='', prior=None, bounds=(-inf, inf)):
        if name in self.names:
            raise ValueError(f'A parameter named "{name}" exists already.')
        super().append(Parameter(name, description, unit, prior, bounds))

    @property
    def names(self):
        return [p.name for p in self]


class ParameterSet(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.blocks = []
        self.bounds = None
        self.frozen = False

    def add_parameter(self, name, description='', unit='', prior=None, bounds=(-inf, inf)):
        if name in self.names:
            raise ValueError(f'A parameter named "{name}" exists already.')
        if not self.frozen:
            super().append(Parameter(name, description, unit, prior, bounds))
        else:
            raise ValueError('Trying to add a parameter to a frozen ParameterSet')

    def add_parameter_block(self, block):
        if block.name in self.blocks:
            raise ValueError(f'A parameter block named "{block.name}" exists already.')
        for p in block:
            if p.name in self.names:
                raise ValueError(f'A parameter named "{p.name}" exists already.')

        s_pre = len(self)
        for p in block:
            super().append(p)
        s_post = len(self)
        setattr(self, f'sl_{block.name.lower()}', slice(s_pre, s_post))
        setattr(self, f'bl_{block.name.lower()}', self[slice(s_pre, s_post)])
        self.blocks.append(block.name)

    def _update(self):
        if not self.frozen:
            self.bounds = array([p.bounds for p in self])
            self.lbounds = self.bounds[:, 0]
            self.ubounds = self.bounds[:, 1]
            for i, p in enumerate(self):
                p.pid = i
        else:
            raise ValueError('Trying to update a frozen ParameterSet')

    def log_prior(self, pv):
        pv = atleast_2d(pv)
        lnp = zeros(pv.shape[0])
        m = all(pv > self.bounds[:, 0], 1) & all(pv < self.bounds[:, 1], 1)
        for i, p in enumerate(self):
            lnp += p.lnprior(pv[:, i])
        return squeeze(where(m, lnp, -inf))

    def freeze(self):
        self._update()
        self.frozen = True

    def thaw(self):
        self.frozen = False

    def find_pid(self, name):
        for p in self:
            if name == p.name:
                return p.pid
        raise KeyError('Could not find parameter {}'.format(name))

    def sample_from_prior(self, size=1):
        return stack([p.rvs(size) for p in self.priors], 1)

    def check_pv(self, pv):
        for i, p in enumerate(self):
            lnp = p.prior.logpdf(pv[i])
            b = self.bounds[i]
            is_finite = p.bounds[0] < pv[i] < p.bounds[1]
            print(f"|{p.pid:3d}| {'*' if not is_finite else ' '} {p.name:10} {b[0]:7.1f} < {pv[i]:10.2f} < {b[1]:<8.1f} {str(p.prior):40} log P = {lnp:6.2f}")

    @property
    def mean_pv(self):
        x0 = zeros(len(self))
        for i, p in enumerate(self.priors):
            if isinstance(p, norm):
                x0[i] = p.mean
            elif isinstance(p, uniform):
                x0[i] = 0.5 * (p.a + p.b)
            else:
                raise ValueError
        return x0

    def set_prior(self, name: str, prior, *nargs):
        pid = self.find_pid(name)
        if hasattr(prior, 'logpdf'):
            self[pid].prior = prior
        elif isinstance(prior, str):
            pname = prior.lower()
            if pname == 'u':
                if len(nargs) != 2:
                    raise ValueError("The uniform prior requires two additional arguments.")
                self[pid].prior = uniform(nargs[0], nargs[1])
            elif pname == 'n':
                if len(nargs) != 2:
                    raise ValueError("The normal prior requires two additional arguments.")
                self[pid].prior = norm(nargs[0], nargs[1])
        else:
            raise ValueError()

    def set_bounds(self, name: str, bounds: tuple[float, float]):
        pid = self.find_pid(name)
        self.thaw()
        self[pid]._bounds = bounds
        self._update()
        self.freeze()

    @property
    def names(self):
        return [p.name for p in self]

    @property
    def units(self):
        return [p.unit for p in self]

    @property
    def descriptions(self):
        return [p.description for p in self]

    @property
    def priors(self):
        return [p.prior for p in self]