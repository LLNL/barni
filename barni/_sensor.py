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
Module containing sensors (detector) definitions.
"""

from . import _architecture as arch
from . import _reader
import numpy as np
from scipy.stats import norm

__all__ = ['GaussianSensorModel']


class GaussianSensorModel(arch.SensorModel):
    ''' Sensor model calculates the response of a detector to incident flux.

    This is a simple detector model relying only on resolution and assumping a
    Gaussian response kernel.

    Attributes:
      resolution : float
        The resolution of the detector, defined as FWHM/energy.
    '''

    def __init__(self, resolution, resolutionEnergy=662,
                 electronicNoise=1, wideningPower=0.6):
        self.resolution = resolution
        self.resolutionEnergy = resolutionEnergy
        self.wideningPower = wideningPower
        self.electronicNoise = electronicNoise
        self._updateCoefficients()

    def toXml(self):
        """ Converts spectrum to XML string

        """

        xml = "<GaussianSensorModel>\n"
        attributes = ["resolution", "resolutionEnergy", "electronicNoise", "wideningPower"]
        for attr in attributes:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if value:
                    xml += "  <%s>" % attr
                    xml += str(value)
                    xml += "</%s>\n" % attr
        xml += "</GaussianSensorModel>\n"
        return xml


    def _updateCoefficients(self):
        fwhmRef_kev = self.resolution * self.resolutionEnergy
        fwhm0_kev = self.electronicNoise
        self.C = self.wideningPower
        self.A = (fwhm0_kev / 2.355)**(1 / self.C)
        self.B = ((fwhmRef_kev / 2.355)**(1. / self.C) -
                  self.A) / self.resolutionEnergy

    def getResponse(self, energy, intensity, binEdges):
        '''
        Integral of gaussian of intensity in channels between bins. Energy
        and intensity have to be the same length

        Args:
          energy(float): is the center of the peak
          intensity(float): is the total counts under the peak
          binEdges(array like): is the edges of the bins
        '''
        scale = self.getResolution(energy)
        binEdges = np.array([binEdges]).T
        u = norm.cdf(binEdges, energy, scale).T
        integral = np.array(intensity * (u[:, 1:] - u[:, :-1]).T).T
        return np.squeeze(integral)
        # FIXME: do the math not max...
        #response = intensity * (integral / integral.max())
        # return response

    def getResponseIntegral(self, energy1, energy2,
                            intensity1, intensity2, binEdges):
        ''' Evaluate the response integral, uses Simpson's Rule '''
        r0 = 0.5 * self.getResolution((energy1 + energy2) / 2)
        n = int(((energy2 - energy1) / r0))
        if (n < 4):
            n = 4
        if (n & 1):
            n += 1
        h = (energy2 - energy1) / n
        out = np.zeros(len(binEdges) - 1)
        # Simpson's Rule
        if (intensity1 != 0):
            out += self.getResponse(energy1, intensity1 * h / 3, binEdges)
        if (intensity2 != 0):
            out += self.getResponse(energy2, intensity2 * h / 3, binEdges)
        # vectorize this
        i = np.arange(1, n)
        q = 2 << (i & 1)
        e = energy1 + i * h
        f = (e - energy1) / (energy2 - energy1)
        f2 = intensity1 * (1 - f) + f * intensity2
        out += self.getResponse(e, f2 * h / 3 * q, binEdges).sum(0)
        return out

    def getResolution(self, energy=662.):
        '''
        Calculates the resolution in units of energy

        This is measured in standard deviations.  Multiply by 2.355 to get FWHM.
        '''
        energy = np.array(energy)
        if (np.any(energy) < 0):
            raise Exception("Energy is negative %f" % energy)
        return (self.A + self.B * energy)**self.C


def loadGaussianSensorModel(context, element):
    '''
    Converts a dom element into a functioning UnfoldingPeakAnalysis object
    '''
    out = GaussianSensorModel(0.08)

    floatFields = [
        'resolution',
        'resolutionEnergy',
        'wideningPower',
        'electronicNoise']
    intFields = []

    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue

        if node.tagName in floatFields:
            out.__setattr__(node.tagName, float(node.firstChild.nodeValue))
            continue

        if node.tagName in intFields:
            out.__setattr__(node.tagName, int(node.firstChild.nodeValue))
            continue

        raise ValueError("Bad tag %s" % node.tagName)

    out._updateCoefficients()

    return out

_reader.registerReader("GaussianSensorModel", loadGaussianSensorModel)
_reader.registerReader("gaussianSensorModel", loadGaussianSensorModel)