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
Flux unfolding module

@author monterial1
"""

from scipy.optimize import nnls
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm
from . import physics


class FluxComponent(ABC):
    ''' Abstract class for components of flux (e.g. continuum, photopeaks) '''

    @abstractmethod
    def getResponsed(self, binEdges, detectorModel):
        ''' Calculates the detector response to the flux component.

        Args:
          binEdges : list, size=N
            The edges between which the response is calculated.
          detectorModel : DetectorModel
            The detector model (e.g. resolution).

        Returns:
          response : list, size=N-1
            Response for each energy bin.
        '''


class Triangle(FluxComponent):
    ''' The continuum representation of the flux component. '''

    def __init__(self, start, middle, end):
        self.start = start
        self.mid = middle
        self.end = end

    def getResponsed(self, binEdges, detectorModel):
        firstHalf = detectorModel.getResponseIntegral(
            self.start, self.mid, 0, 1, binEdges)
        secondHalf = detectorModel.getResponseIntegral(
            self.mid, self.end, 1, 0, binEdges)
        return firstHalf + secondHalf

    def __str__(self):
        return "continuum-triangle(start=%f, end=%f)" % (self.start, self.end)


class HalfTriangle(FluxComponent):
    ''' Forbidden zone filler '''

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def getResponsed(self, binEdges, detectorModel):
        firstHalf = detectorModel.getResponseIntegral(
            self.start, self.end, 0, 1, binEdges)
        return firstHalf


# Component for low angle scatter
class Scattering(FluxComponent):
    ''' Forbidden zone filler '''

    def __init__(self, energy):
        edge = physics.computeEdge(energy)

        self.start = np.maximum(edge - (energy - edge) / 2, 10)
        self.mid = edge
        self.end = energy
        if (self.start < 0):
            raise Exception("Bad scattering %f,%f" % (energy, edge))

    def getResponsed(self, binEdges, detectorModel):
        part1 = detectorModel.getResponseIntegral(
            self.start, self.mid, 0, 0.4, binEdges)
        part2 = detectorModel.getResponseIntegral(
            self.mid, self.end, 0.4, 1, binEdges)
        return part1 + part2

    def __str__(self):
        return "scattering(energy=%f)" % (self.end)


class Line(FluxComponent):
    ''' The discrete energy representation of the flux component '''

    def __init__(self, center):
        self.center = center

    def getResponsed(self, binEdges, detectorModel):
        return detectorModel.getResponse(self.center, 1, binEdges)

    def __str__(self):
        return "line(energy=%f)" % (self.center)
