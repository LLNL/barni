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
'''
This module is for extracting peaks using a smooth continuum assumption.

This is a faster method to unfold a spectrum which makes the assumption
that the baseline is a smooth continuous curve which is wider than the
expected response kernel.  It does a series of passes to get the approximate
baseline, then computes the derivative of the residual to find an initial
set of peaks, then unfolds the spectrum onto the assumed peak list and
a continous smooth curve.

@author monterial1
'''

import numpy as np
import math
import textwrap

from typing import List

from ._architecture import IdentificationInput, PeakResults
from ._reader import registerReader
from . import _architecture as arch
from ._spectrum import Spectrum
from ._smoothing import smooth
from ._math import SolveAugmentedTridiag, gauss_pdf

__all__ = ['SmoothPeakAnalysis', 'computeBaseline']


def computeBaseline(y, mu=1):
    """ Function to get the baseline for initial peak scan.

    Args:
        y (array-like): Spectra to extract baseline from.
        mu (float): Smoothing kernel.

    Returns:
        Baseline estimate and the slightly smoothed initial spectra.
    """

    u = smooth(y, lambda p: p * mu * 0.05)
    x = smooth(u, lambda p: p * mu)
    for i in range(0, 2):
        mu = mu / 2
        x = np.maximum(x, 0)
        x = smooth(np.minimum(u - x, 0), lambda p: p * mu) + x
    x = np.maximum(x, 0)
    return x, u


def getInitialPeaks(y, b, es, sensor=None, lld=45, mu=1):
    """ Function to get estimating initial peaks.

       This is based on derivate based method of scanning for location inflection points.
    """
    #
    u = smooth(y, lambda p: p * mu * 0.05)
    s = u - b
    potential = []
    current = u[0] - b[0]
    prev = current
    rising = False

    for i in range(0, len(u)):
        nxt = u[i] - b[i]

        # ignore flats
        if (current == nxt):
            prev = current
            continue

        # tag maxima
        if (rising and current > nxt):
            rising = False
            sig = current / np.sqrt(np.maximum(b[i - 1], 1))
            if (sig > 1):
                # Use interpolation to get the likely center
                p1 = np.maximum((nxt + current) / 2, 0)
                p2 = np.maximum((prev + current) / 2, 0)
                f = 0.5 * (p2 - p1) / (p2 + p1)

                # Compute the initial peak parameters
                channel = i - 1 + f
                energy = es[channel]
                intensity = current

                # Add it to the list
                out = arch.Peak(energy, intensity, 0)
                out.channel = channel
                potential.append(out)

        # catch falling
        if (not rising and current < nxt):
            rising = True

        prev = current
        current = nxt

    # Find closely spaced peaks and merge them
    peaks = []
    i = 0
    while i < len(potential):
        p1 = potential[i]
        e0 = p1.energy
        if e0 < lld:
            i = i + 1
            continue
        e = e0 + sensor.getResolution(e0) * 2.35
        j = i + 1

        while j < len(potential) and potential[j].energy < e:
            # Merge based on intensity
            p2 = potential[i + 1]
            f = p1.intensity / (p1.intensity + p2.intensity)

            # Create the merged peak
            channel = f * p1.channel + (1 - f) * p2.channel
            energy = f * p1.energy + (1 - f) * p2.energy
            intensity = f * p1.intensity + (1 - f) * p2.intensity
            p1 = arch.Peak(energy, intensity, 0)
            p1.channel = channel

            # Update the end point
            e0 = energy
            e = e0 + sensor.getResolution(e0) * 2.35
            j = j + 1
        peaks.append(p1)
        i = j

    return peaks


def responsePeaks(peaks, sensor, energyScale):
    """ Augment the peak list based on the sensor response
    Args:
        peaks: List of found peaks.
        sensor: Sensor model used for responding the peaks.
        energyScale: Energy scale.

    Returns:
        List of responsed peaks.

    """

    for i in range(0, len(peaks)):
        p = peaks[i]
        p.response = sensor.getResponse(p.energy, 1, energyScale)
    return peaks

def solve(spectrum, peaks, sensor, es, mu=1, lld=0):
    """ Simultenously solves the smooth curve and peaks.

    Args:
        spectrum: Input spectrum.
        peaks: Initially found peaks, respponsed through the sensor.
        sensor: Sensor model used to response peaks.
        es: Energy scale.
        mu: Smoothing parameter
        lld: In channel space

    Returns:
        Estimate of the baseline, intensity, peaks and response.
    """
    # FIXME: this can be done much better with a specialized solver.

    # Collect the peak shapes from the peak list

    shapes = [i.response for i in peaks]
    n1 = len(spectrum.counts)
    n2 = len(shapes)
    if not peaks:
        S = np.zeros((n1,1))
    else:
        S = np.array(shapes).T
    # Allocate space for problem
    n = n1 + n2
    A = np.zeros((n, n))
    B = np.zeros((n, 1))
    # Set up the tridiagonal portion
    c2 = 0
    c = 1
    for i in range(0, n1 - 1):
        if i <= lld:
            c = 0
        else:
            c = i * mu
        A[i, i] = 1 + c + c2
        A[i, i + 1] = -c
        A[i + 1, i] = -c
        c2 = c
    A[i, i] = 1 - c2
    # Populate the unfolding portion
    B[0:n1] = np.array([spectrum.counts]).T
    B[n1:] = S.T @ B[0:n1]
    A[n1:, 0:n1] = S.T
    A[0:n1, n1:] = S
    A[n1:, n1:] = S.T @ S
    # turn the triag into a diag with 1 in main diag
    A11 = A[:n1,:n1]
    A12 = A[:n1,n1:]
    A21 = A[n1:,:n1]
    A22 = A[n1:,n1:]
    B1  = B[:n1]
    B2  = B[n1:]
    solver = SolveAugmentedTridiag(A11, A12, A21, A22, B1, B2)
    solver.solve()
    # solve for baseline
    baseline = solver.C1
    intensity = solver.C2
    # Strip off all but the required information
    peaksOut = []
    for i, pk in enumerate(peaks):
        # FIXME we may want to compute the baseline here ?
        width = sensor.getResolution(pk.energy)
        peaksOut.append(arch.Peak(pk.energy, float(intensity[i]), 0, width))
    return baseline, intensity, peaksOut, S


class SmoothPeakResult(arch.PeakResult):
    """ Smooth peak analysis implementation of a peak result.

    Attributes:
        peaks (PeakList): List of peaks found.
        continuum (Spectrum): Estimated continuum.
        sensor (SensorModel): The sesnor model used to response the peaks.
    """
    def __init__(self, peaks : List[arch.Peak], continuum : Spectrum, sensor : arch.SensorModel):
        self._peaks = peaks
        self._continuum = continuum
        self._sensor = sensor

    def getContinuum(self) -> Spectrum:
        return self._continuum

    def toXml(self, name = None):
        """
        Args:
            name: Attribute to tag the peaks result with
        """
        if name is None:
            xml = "<SmoothPeakResult>\n"
        else:
            xml = "<SmoothPeakResult name='%s'>\n" % name
        for peak in self.getPeaks():
            xml += textwrap.indent(peak.toXml(), "  ")
        xml += textwrap.indent(self._continuum.toXml("continuum"), "  ")
        xml += textwrap.indent(self._sensor.toXml(), "  ")
        xml += "</SmoothPeakResult>\n"
        return xml

    def getPeaks(self) -> List[arch.Peak]:
        return self._peaks

    def getRegionOfInterest(self, roi: arch.RegionOfInterest) -> arch.Peak:
        e1 = roi.lower
        e2 = roi.upper

        # Filter the peak list to get the total in the roi
        intensity = 0
        energy = 0
        root2 = np.sqrt(2)
        for peak in self._peaks:
            if peak.energy > e2 and (peak.energy - e2) / peak.width > 4:
                continue
            if peak.energy < e1 and (e1 - peak.energy) / peak.width > 4:
                continue
            # integrate the region of interest
            t2 = math.erf((e2 - peak.energy)/peak.width/root2)
            t1 = math.erf((e1 - peak.energy)/ peak.width/root2)
            contribution = (t2 - t1) * peak.intensity / 2.
            intensity += contribution
            # effective energy of the region of interest
            energy += peak.intensity * (
                       peak.energy / 2 * (t2 - t1)
                      - peak.width**2 *(gauss_pdf(e2, peak.energy, peak.width) - gauss_pdf(e1, peak.energy, peak.width)))
        '''    
        for peak in [i for i in self._peaks if (i.energy > e1 and i.energy < e2)]:
            intensity += peak.intensity
            energy += peak.energy * peak.intensity
        '''
        if (intensity > 0):
            energy /= intensity

        # Compute the baseline as the sum accross the region of the continuum
        if self._continuum:
            baseline = np.max([0, self._continuum.getIntegral(e1, e2)])
        else:
            baseline = 0.

        # Get the total over the roi
        return arch.Peak(energy, intensity, baseline)

    def getEnergyScale(self):
        return self._continuum.energyScale

    def getFit(self):
        """ Returns the fit estimate.
        """
        out = np.array(self._continuum.counts).flatten()
        energyScale = self._continuum.energyScale
        for p in self._peaks:
            shape = self._sensor.getResponse(
                p.energy, float(p.intensity), energyScale)
            out = out + shape
        return Spectrum(out, energyScale)


class SmoothPeakAnalysis(arch.PeakAnalysis):
    """ Smooth peak analysis implementation of the peak extractor.

    This method uses a combination of a smoothing method and deriviative peak finder to
    extract both the peaks and provide the estimation of the continuum

    Attributes:
        sensor: Sensor model used to response the found peaks.
        startEnergy: Energy to begin the peak search.
        endEnergy: Energy to end the peak search.
        smoothingFactor: Smoothing factor to use for analysis.

    """

    def __init__(self):
        self.sensor = None
        self.startEnergy = 35
        self.endEnergy = 3000
        self.smoothingFactor = 3

    def __analyze_spectrum(self, spectrum : Spectrum) -> arch.PeakResult:
        """ Performs peak extraction on the inputed spectrum

        Args:
            spectrum (Spectrum)

        Returns:
            PeakResult container of the extracted peaks.
        """

        energyScale = spectrum.energyScale

        # Smoothing kernel is based on the average energy scale
        mu = self.smoothingFactor * \
            len(energyScale) / (energyScale[-1] - energyScale[0])

        # Compute the initial baseline to remove low frequency components
        baseline0, y = computeBaseline(spectrum.counts, mu=mu)

        # Extract an initial peak list used the derivative
        peaks0 = getInitialPeaks(
            y, baseline0, energyScale, sensor=self.sensor, lld=self.startEnergy, mu=mu)

        # Compute the response kernel for the initial peaks
        peaks0 = responsePeaks(peaks0, self.sensor, energyScale)

        # Solve the smooth curve plus peaks
        baseline, intensity, peaks, response = solve(
            spectrum, peaks0, self.sensor, energyScale, mu=mu, lld = energyScale.findBin(self.startEnergy))

        # Produce a standard output
        continuum  = Spectrum(np.array(baseline).flatten(), energyScale)
        for peak in peaks:
            peak.baseline = continuum.getIntegral(peak.energy-peak.width, peak.energy+peak.width)
        out = SmoothPeakResult(peaks, continuum, self.sensor)
        return out

    def analyze(self, id_input: IdentificationInput, downsample = False) -> PeakResults:
        """
        Args:
            id_input (IdentificationInput): BARNI identification input.
            downsample (default, False): Optional for downsampling the inputs.

        Returns:
            Container of all the peak results included in the identification input.

        Notes:
            We are downsampling for speed currently but this needs to depend
            on the input energy scale.

            We should likely use a proper rebinning function so that we aren't
            affected by the input bin structure directly.
        """

        if downsample:
            sample = id_input.sample.downsample()
        else:
            sample = id_input.sample
        sample_result = self.__analyze_spectrum(sample)
        intrinsic_result = None
        scale_factor = None
        if hasattr(id_input, 'intrinsic'):
            if id_input.intrinsic is not None:
                if downsample:
                    intrinsic = id_input.intrinsic.downsample()
                else:
                    intrinsic = id_input.intrinsic
                intrinsic_result = self.__analyze_spectrum(intrinsic)
                scale_factor = sample.livetime * 1. / intrinsic.livetime
        out = PeakResults(sample_result, intrinsic_result, scale_factor)
        return out

def loadSmoothPeakAnalysis(context, element):
    '''
    Converts a dom element into a functioning UnfoldingPeakAnalysis object
    '''
    out = SmoothPeakAnalysis()

    floatFields = [
        'startEnergy',
        'endEnergy',
        'smoothingFactor',
    ]
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


registerReader("smoothPeakAnalysis", loadSmoothPeakAnalysis)