from dataclasses import dataclass
from typing import List


@dataclass
class LightCurveData:
    time: List            # List of time arrays
    flux: List            # List of flux arrays
    covariates: List      # List of covariate arrays
    passband: List        # Passband per light curve
    noise: List           # White noise estimate per light curve
    instrument: List      # Instrument name per light curve
    sector: List          # Sector or similar id per light curve
    segment: List         # Segment id per light curve
    exptime: List         # Exposure time per light curve
    nsamples: List        # Number of supersamples per light curve

    def __add__(self, other):
        return LightCurveData(self.time+other.time, self.flux+other.flux, self.covariates+other.covariates,
                          self.passband+other.passband, self.noise+other.noise, self.instrument+other.instrument,
                          self.sector+other.sector, self.segment+other.segment,
                          self.exptime+other.exptime, self.nsamples+other.nsamples)