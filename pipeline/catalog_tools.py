# an assortment of tools that relate to getting and manipulating source catalogs

import numpy as np


class Bandpass:
    """A helper class to keep track of the bandpass of filters.

    This is currently only used for determining which filter in the instrument
    is most similar to a filter in a reference catalog.
    For example, is a V filter closer to Gaia G or to Gaia B_P or R_P?
    """
    def __init__(self, *args):
        """Create a new Bandpass object.

        Currently, initialize using lower/upper wavelength in nm.
        This assumes all filters have a simple, top-hat bandpass.
        This is good enough for comparing a discrete list of filters
        (e.g., the Gaia filters vs. the DECam filters) but not for
        more complicated things like finding the flux from a spectrum.

        Additional initializations could be possible in the future.

        """
        self.lower_wavelength = args[0]
        self.upper_wavelength = args[1]

    def __getitem__(self, idx):
        """A shortcut for getting the lower/upper wavelength.
        band[0] gives the lower wavelength, band[1] gives the upper wavelength.
        """
        if idx == 0:
            return self.lower_wavelength
        elif idx == 1:
            return self.upper_wavelength
        else:
            raise IndexError("Bandpass index must be 0 or 1. ")

    def get_overlap(self, other):
        """Find the overlap between two bandpasses.

        Will accept another Bandpass object, or a tuple of (lower, upper) wavelengths.
        By definition, all wavelengths are given in nm, but this works just as well
        if all the bandpasses are defined in other units (as long as they are consistent!).

        """

        return max(0, min(self.upper_wavelength, other[1]) - max(self.lower_wavelength, other[0]))

    def get_score(self, other):
        """Find the score (goodness) of the overlap between two bandpasses.

        The input can be another Bandpass object, or a tuple of (lower, upper) wavelengths.
        The score is calculated by the overlap between the two bandpasses,
        divided by the sqrt of the width of the other bandpass.
        This means that if two filters have similar overlap with this filter,
        the one with a much wider total bandpass gets a lower score.

        This can happen when one potentially matching filter is very wide
        and another potential filter covers only part of it (as in Gaia G,
        which is very broad) but the narrower filter has good overlap,
        so it is a better match.
        """
        width = other[1] - other[0]
        return self.get_overlap(other) / np.sqrt(width)

    def find_best_match(self, bandpass_dict):
        """Find the best match for this bandpass in a dictionary of bandpasses.

        If dict is empty or all bandpasses have a zero match to this bandpass,
        returns None.

        # TODO: should we have another way to match based on the "nearest central wavelength"
           or somthing like that? Seems like a corner case and not a very interesting one.
        """
        best_match = None
        best_score = 0
        for k, v in bandpass_dict.items():
            score = self.get_score(v)
            if score > best_score:
                best_match = k
                best_score = score

        return best_match
