from __future__ import division

import logging

import numpy as np
from astropy.nddata import NDDataRef
from astropy.wcs import WCS, WCSSUB_SPECTRAL
from astropy.units import Unit, Quantity
from astropy import units as u

from .spectrum_mixin import OneDSpectrumMixin

__all__ = ['Spectrum1D']


class Spectrum1D(OneDSpectrumMixin, NDDataRef):
    """
    Spectrum container for 1D spectral data.
    """
    def __init__(self, flux, spectral_axis=None, wcs=None, unit=None,
                 spectral_axis_unit=None, *args, **kwargs):

        if not isinstance(flux, Quantity):
            flux = Quantity(flux, unit=unit or "Jy")

        if spectral_axis is not None and not isinstance(spectral_axis, Quantity):
            spectral_axis = Quantity(spectral_axis, unit=spectral_axis_unit or u.AA)

        # If spectral_axis had not been defined, attempt to use the wcs
        # information, if it exists
        if spectral_axis is None and wcs is not None:
            if isinstance(wcs, WCS):
                # Try to reference the spectral axis
                wcs_spec = wcs.sub([WCSSUB_SPECTRAL])

                # Check to see if it actually is a real coordinate description
                if wcs_spec.naxis == 0:
                    # It's not real, so attempt to get the spectral axis by
                    # specifying axis by integer
                    wcs_spec = wcs.sub([wcs.naxis])

                # Construct the spectral_axis array
                spectral_axis = wcs_spec.all_pix2world(
                    np.arange(flux.shape[0]), 0)[0]

                # Try to get the spectral_axis unit information
                try:
                    spectral_axis_unit = wcs.wcs.cunit[0]
                except AttributeError:
                    logging.warning("No spectral_axis unit information in WCS.")
                    spectral_axis_unit = Unit("")

                spectral_axis = spectral_axis * spectral_axis_unit

                if wcs.wcs.restfrq != 0:
                    self._rest_value = wcs.wcs.restfrq * u.Hz
                elif wcs.wcs.restwav != 0:
                    self._rest_value = wcs.wcs.restwav * u.AA

        super(Spectrum1D, self).__init__(data=flux.value, unit=flux.unit,
                                         wcs=wcs, *args, **kwargs)



    @property
    def frequency(self):
        return self.spectral_axis.to(u.GHz, u.spectral())

    @property
    def wavelength(self):
        return self.spectral_axis.to(u.AA, u.spectral())

    @property
    def energy(self):
        return self.spectral_axis.to(u.ev, u.spectral())
