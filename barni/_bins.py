###############################################################################
# Copyright (c) 2019 Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
#
# Written by M. Monterial, K. Nelson
# monterial1@llnl.gov
#
# LLNL-CODE-<R&R number>
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

"""
Module for holding the concept of an energy scale.

Spectra are defined as histograms. These histograms have bins.  Each channel is defined as having an upper and lower energy edge.
There are one more edges than bins.  Functions in this module provide utilities for working with these edges.
"""

import numpy as np
import bisect

from ._reader import registerReader
from ._architecture import Serializable

__all__ = ['EnergyScale']


class EnergyScale(Serializable):
    ''' Energy scale holds the binning information of spectra.

        Args:
            edges (array_like): The edges of the bins.
    '''

    def __init__(self, edges):
        self._edges = np.array(edges)

    @staticmethod
    def newScale(start, end, startStep, endStep):
        ''' Produce a new energy scale given a start and end range with a initial bin width and a final bin width.

        This computes the optimum number of bins to achieve the desired spacing.

        We don't always deal with detectors for which a quadratic scale is the best
        solution to binning.  Instead we will give the desired FWHM at each end of
        the scale and compute the binning structure to cover the space.

        Args:
            start (int): Start of the first bin.
            end (int): End of the last bin.
            startStep (int): Initial bin width.
            endStep (int): Final bin width.

        Returns:
            EnergyScale: New energy scale created from input parameters.

        '''
        # Guess how many steps it will take to cover the space
        n0 = (end - start) / startStep
        n1 = (end - start) / endStep

        # Find integer bins that fit in the scale
        n = int((n0 + n1) / 2)
        while True:
            accel = (endStep - startStep) / (n - 1)
            g = start + startStep * n + accel * (n - 1) * (n) / 2
            if (g < end):
                break
            n = n - 1

        # Compute much we miss the final point by
        miss0 = (end - g) / n

        # Consider the alternative structure with one additional bin
        accel1 = (endStep - startStep) / n
        g1 = start + startStep * (n + 1) + accel1 * (n) * (n + 1) / 2
        miss1 = (end - g1) / (n + 1)

        # Whichever structure is closest is the best
        if miss0 > -miss1:
            accel = accel1
            m0 = startStep
            n = n + 1
        else:
            m0 = startStep + miss0

        # Compute the final scale
        edges = np.zeros((n + 1,))
        x = start
        m = m0
        for i in range(0, n + 1):
            edges[i] = start + m0 * i + accel * (i - 1) * (i) / 2

        return EnergyScale(edges)

    """ Get which bin this energy falls into"""

    def findBin(self, energy):
        ''' Finds bin number corresponding to a specific energy.
        '''
        return bisect.bisect_left(self._edges, energy) - 1

    def getCenter(self, i):
        ''' Returns centers at specified bin number.
        '''
        return (self._edges[i] + self._edges[i + 1]) / 2

    def __len__(self):
        return len(self._edges)

    # def __getitem__(self, i):
    #    return self._edges[i]

    def findEnergy(self, i):
        j = int(i)
        f = i - j
        return self._edges[j] * (1 - f) + f * self._edges[j + 1]

    def getCenters(self):
        '''
        Returns:
            centers (array_like): Centers between the bin edges.
        '''
        return (self._edges[0:-1] + self._edges[1:]) / 2

    def getEdges(self):
        return self._edges

    def toXml(self):
        """ XML representation of the energy scale class
        """
        xml = "<EnergyScale>\n"
        xml += "  <edges>"
        for edge in self._edges:
            xml += str(edge) + " "
        xml += "</edges>\n"
        xml += "</EnergyScale>\n"
        return xml

def loadEnergyScale(context, element):
    """
    Converts a dom element into a FeatureExtractor object
    """
    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName == "edges":
            edges = np.fromstring(node.firstChild.data, sep=" ")
            continue
        context.raiseElementError(element, node)
    out = EnergyScale(edges)
    return out

registerReader("EnergyScale", loadEnergyScale)

