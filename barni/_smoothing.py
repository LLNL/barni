###############################################################################
# Copyright (c) 2019 Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
#
# Written by M. Monterial, K. Nelson
# monterial1@llnl.gov
#
# LLNL-CODE-805904
#
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED,INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
###############################################################################
'''
Collection of smoothing tools.

@author monterial1
'''

import numpy as np
from scipy.linalg import solve_banded
import _barni

__all__ = ['smooth']


def smooth(signal, lmbda):
    """ Preform smoothing on a spectrum with a variable
    width contraint.

    This filter applies a cusp like filter to data.
    The smoothing can vary by channel resulting smoothing
    different parts.  Using a linear smoothing constraint
    will smooth the data with peak withs varying according
    to Poisson statistics properly.

    The method uses a linear algebra problem in which we
    jointly try to match the observed data and the requirement
    that the derivate by minimum.

    This procedure does not produce an output with the
    same integral as the original except in cases where the
    constraint is constant.

    Args:
        signal (array-like): Spectrum to be smoothed.
        lmbda (float, lambda): The smoothing factor to apply across the energy spectrum.
            If it is a constant then a uniform smoothing is applied across all bins. For
            gamma-ray spectra a lambda expression that varies linearly with the energy
            scale will ensure that the smoothing factor is appropriately widened as a
            function of energy, following Poisson statistic from the collected charge
            carriers.

    Returns:
        Smoothed version of the signal array.

    Notes:
        Tridiagonal matrix solving is about the fastest of all the matrix problems one can come up with
        but Python still managed to run much slower than one should expect. The cost of simply doing
        the forward and backsubstitution was eating about 30 ms for a very reasonably sized problem.
        I have thus replaced it with the specialized banded solver from scipy linear algebra.  This cut
        the time by a factor of 30.  We can save another factor of 2 if we vectorize the generation of
        problem, but it isn't worth doing at this point.

    """
    n = len(signal)
    A = np.zeros((3, n))
    _barni.fill_smooth(A, lmbda)
    return solve_banded((1, 1), A, np.array(signal, np.float))

