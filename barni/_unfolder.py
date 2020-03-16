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
This module is for unfolding spectra, extracting the underlying flux at the
face of the detector.

@author monterial1
'''

import numpy as np
from scipy.optimize import nnls
import textwrap
from typing import Dict, List

from . import _flux
from . import _reader
from . import _architecture as arch
from . import physics
from ._spectrum import Spectrum
from ._smoothing import smooth

__all__ = ['UnfoldingPeakAnalysis']


def createPickPoints(start, stop, steps):
    ''' Creates pick points on a sqrt scale '''
    return np.linspace(np.sqrt(start), np.sqrt(stop), steps)**2


def points2triangles(points, span=2):
    ''' Converts list of pick points to triangles. '''
    triangles = []
    for i in range(span, len(points) - span):
        triangles.append(
            _flux.Triangle(points[i - span], points[i], points[i + span]))
    return triangles


class UnfoldingResult(object):
    """
    (internal) auditing result from the unfolder
    """

    def getFitContinuum(self, edges):
        A = np.zeros((len(edges) - 1))
        for component in self._continuumComponents:
            A = A + component.getResponsed(edges,
                                           self._sensor) * component.intensity
        return A

    def getFitScattering(self, edges):
        A = np.zeros((len(edges) - 1))
        for component in self._scatteringComponents:
            A = A + component.getResponsed(edges,
                                           self._sensor) * component.intensity
        return A

    def getFitLines(self, edges):
        A = np.zeros((len(edges) - 1))
        for component in self._lineComponents:
            A = A + component.getResponsed(edges,
                                           self._sensor) * component.intensity
        return A

#        idx = [i for i, j in enumerate(
#            fluxComponents) if isinstance(j, _flux.Triangle)]
#        result._fit_continuum = np.matmul(Am[:, idx], x[idx])

#        idx = [i for i, j in enumerate(
#            fluxComponents) if isinstance(j, _flux.Scattering)]
#        result._fit_scattering = np.matmul(Am[:, idx], x[idx])

#        idx = [i for i, j in enumerate(
#            fluxComponents) if isinstance(j, _flux.Line)]
#        result._fit_lines = np.matmul(Am[:, idx], x[idx])


class UnfoldingPeakResult(arch.PeakResult):
    '''
    Result for an unfolding peak analysis when used as a PeakAnalysis object.
    '''

    def __init__(self):
        self._sampleAnalysis = None
        self._peaks = []

    def getBaseline(self):
        """ Lacks baseline estimate
        """
        return None

    def getFit(self):
        """ Returns the best fit to the spectrum.
        """

    def toXml(self, name=None):
        """
        Args:
            name: Attribute to tag the peaks result with
        """
        if name is None:
            xml = "<UnfoldingPeakResult>\n"
        else:
            xml = "<UnfoldingPeakResult name='%s'>\n" % name
        for peak in self.getPeaks():
            xml += textwrap.indent(peak.toXml(), "  ")
        xml += "</UnfoldingPeakResult>\n"
        return xml

    def getPeaks(self) -> List[arch.Peak]:
        # if we have cached it already, then return the cache
        if len(self._peaks) > 0:
            return self._peaks

        # Otherwise compute it
        peaks = []
        sa = self._sampleAnalysis
        for line in sa._lineComponents:
            if line.intensity == 0:
                continue

            # Compute the FWHM
            res = 2.355 * sa._sensor.getResolution(line.center)
            e1 = line.center - res / 2
            e2 = line.center + res / 2

            # Compute the intensity
            intensity = line.getResponsed(
                [e1, e2], sa._sensor) * line.intensity

            # Compute the baseline
            baseline = 0
            #  Computing the baseline is exceptionally slow.
            #    FIXME we are going to need to cache some information computed in
            #    the earlier step to make this possible.
#            for cont in sa._continuumComponents:
#                baseline += cont.getResponsed([e1, e2], sa._sensor) * cont.intensity

            # Store as a peak
            peak = arch.Peak(line.center, intensity, baseline)
            peaks.append(peak)

        self._peaks = peaks
        return peaks

    def getRegionOfInterest(self, roi: arch.RegionOfInterest) -> arch.Peak:
        e1 = roi.lower
        e2 = roi.upper

        sa = self._sampleAnalysis

        # Filter the peak list to get the total in the roi
        intensity = 0
        energy = 0
        for peak in [i for i in sa._lineComponents if (
                i.center > e1 and i.center < e2)]:
            intensity += peak.intensity
            energy += peak.center * peak.intensity

        if (intensity > 0):
            energy /= intensity

        # Use the audit log to find baseline
        baseline = 0
#        for cont in sa._continuumComponents:
#            baseline += cont.getResponsed([e1, e2],
#                                          sa._sensor) * cont.intensity

        # Get the total over the roi
        return arch.Peak(energy, intensity, baseline)


class UnfoldingPeakAnalysis(arch.PeakAnalysis):
    def __init__(self):
        self.sensor = None
        self.numPoints = 50
        self.triangleSpan = 2
        self.startEnergy = 0
        self.endEnergy = 3000
        self.toleranceOffset = 1
        self.threshold = 2
        self.maxLinesAdded = 50
        self.smoothingCoef = 0

    def analyze(self, idInput: arch.IdentificationInput) -> arch.PeakResult:
        '''
        Front end to produce a generic peak analysis result.
        '''
        out = UnfoldingPeakResult()

        # Unfold the spectrum to get the peaks
        out._sampleAnalysis = self.unfold(idInput.sample)

        # We could unfold the background as well and used the background
        # to cut down peaks that do not belong to the source or
        # refine our baseline estimate (as the baseline can't be lower
        # that background), but that work is for later.

        return out

    def unfold(self, sample: Spectrum):

        sampleSpectrum = np.array(sample.counts)
        binEdges = sample.energyScale.getEdges()

        # We want equal sensitivity to high energy and low energy peaks, but
        # high energy peaks tend to be much lower amplitude to the peak
        # width and the changes in detector efficiency.  To deal with this
        # issue we will normalize the residuals so that high energy peaks have
        # a lower effective threshold than low energy peaks.
        binCenters = sample.energyScale.getCenters()
        importance = np.sqrt(binCenters / 1000)

        # FIXME this is tuned towards our current training set
        # binning structure.  It would need depend on the
        # detector resolution and the energy binning in practice
        if self.smoothingCoef:
            smoothedSpectrum = smooth(
                sampleSpectrum,  self.smoothingCoef)
        else:
            smoothedSpectrum = sampleSpectrum

        # this is the master loop
        # put together the regressor matrix
        # maybe split between lines and continuum
        fluxComponents = []
        pickPoints = createPickPoints(
            self.startEnergy, self.endEnergy, self.numPoints)
        fluxComponents.extend(points2triangles(pickPoints, self.triangleSpan))

        # setup the problem, the regression coefficients
        A = []  # this will hold the columns of responsed components FluxComponents
        added = 0
        for component in fluxComponents:
            # bin edges are used for the integration, detectorResponse holds
            # gaussians
            A.append(component.getResponsed(binEdges, self.sensor))

        # Tolerance is a vector of how much deviation we will tolerate by
        # energy bin
        tolerance = np.sqrt(sampleSpectrum + self.toleranceOffset)

        # At the start of the problem we will try to fit with nothing but broad
        # continuum only shapes.  Certain portions of the spectrum will fit
        # poorly because the peaks are too narrow to fit with broad shapes.
        # These narrow features will be added one at a time into the fit until the entire
        # spectrum is fit.

        while (True):
            Am = np.array(A).T
            # Am holds the shape and x is the intensity
            x, err = nnls(Am, sampleSpectrum)
            y_hat = np.matmul(Am, x)

            # We use the residuals to consider if there is a peak
            # that can't be fit using the current assumptions regarding
            # continuum.
            residuals = (smoothedSpectrum - y_hat) / tolerance
            residuals = np.maximum(residuals * importance, 0)

            ind = residuals.argmax()
            print("Consider", residuals[ind])
            if (residuals[ind] < self.threshold or added > self.maxLinesAdded):
                break
            tolerance[ind] *= 1000
            added += 1

#            en = binCenters[ind]  # Use the exact bin center
            # FIXME the next line should be placing the lines such that the do not
            #  correspond to the exact position of a bin center, but we are seeing lots
            #  of stripping in the output that indicates that it may not be working properly.
            #  perhaps another approach is required to get the lines well placed.
            #
            #  If it were not for closely spaced lines we could do a regional fit of a Gaussian
            #  and a line which would give us a good start, but this would take time it get
            #  done properly when we consider a neighboring peak.
            #
            #  Alternatively we could apply slope based method to find the optimium inflection point,
            #  but that would suffer issues with low count data.
            en = (binCenters[ind - 1] * residuals[ind - 1] + binCenters[ind] * residuals[ind] +
                  binCenters[ind + 1] * residuals[ind + 1]) / np.sum(residuals[ind - 1:ind + 2])
            if en < 0:
                print("Error in energy calculation")
                print(en)
                print(binCenters[ind])
                print(residuals[ind - 1:ind + 2])
                raise RuntimeError("Negative energy")

            # Add a new line
            line = _flux.Line(en)
            fluxComponents.append(line)
            A.append(line.getResponsed(binEdges, self.sensor))

            # Add a new scattering component
            #   We need to impose 2 conditions on adding this.
            #   1) The line we are extracting must be strong relative to the
            #      continuum we are extracting the peak from.
            #   2) The spacing between the Compton edge and the peak must
            #      be sufficiently far apart that we can see the difference.
            #  This must not be in terms of channels as we may work the problem
            # at different resolutions, but rather in terms of fwhm of the
            # detector.
            edge = physics.computeEdge(en)
            if (en - edge) > 3 * self.sensor.getResolution(en):
                scattering = _flux.Scattering(en)
                fluxComponents.append(scattering)
                A.append(scattering.getResponsed(binEdges, self.sensor))

        # Separate the components by type.
        for i in range(0, len(fluxComponents)):
            fluxComponents[i].intensity = x[i]

        # Eliminate zero components.
        fluxComponents = [f for f in fluxComponents if f.intensity > 0]

        # Store all of our working space in the result for auditing (as private
        # members)
        result = UnfoldingResult()
        result._pickPoints = pickPoints
        result._continuumComponents = [
            i for i in fluxComponents if isinstance(i, _flux.Triangle)]
        result._lineComponents = [
            i for i in fluxComponents if isinstance(i, _flux.Line)]
        result._scatteringComponents = [
            i for i in fluxComponents if isinstance(i, _flux.Scattering)]
        result._sensor = self.sensor
#        result._fluxComponents = fluxComponents
#        result._A = A
#        result._x = x


#        idx = [i for i, j in enumerate(
#            fluxComponents) if isinstance(j, _flux.Triangle)]
#        result._fit_continuum = np.matmul(Am[:, idx], x[idx])

#        idx = [i for i, j in enumerate(
#            fluxComponents) if isinstance(j, _flux.Scattering)]
#        result._fit_scattering = np.matmul(Am[:, idx], x[idx])

#        idx = [i for i, j in enumerate(
#            fluxComponents) if isinstance(j, _flux.Line)]
#        result._fit_lines = np.matmul(Am[:, idx], x[idx])

        # Compute our peaks for the feature extractor
        return result


def loadUnfoldingPeakAnalysis(context, element):
    '''
    Converts a dom element into a functioning UnfoldingPeakAnalysis object
    '''
    out = UnfoldingPeakAnalysis()

    floatFields = [
        'startEnergy',
        'endEnergy',
        'threshold',
        'toleranceOffset',
        'smoothingCoef']
    intFields = ['maxLinesAdded', 'numPoints', 'triangleSpan']

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

        if node.tagName == "sensor":
            children = [
                i for i in node.childNodes if i.nodeType == node.ELEMENT_NODE]
            sensor = context.convert(children[0])
            if not isinstance(sensor, arch.SensorModel):
                raise TypeError("Bad sensor model")
            out.__setattr__("sensor", sensor)
            continue

        raise ValueError("Bad tag %s" % node.tagName)

    return out


_reader.registerReader("unfoldingPeakAnalysis", loadUnfoldingPeakAnalysis)
