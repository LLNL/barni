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

"""
Region of interest module.

This code is tasked with taking a list of lines that we have extracted from a
collection of spectra from a nuclide with differing shielding configurations
to create a set of regions that are indicative of each nuclide.

The regions of interest will be used to defined the feature extractor which
converts the peak extraction into a fixed vector which is passed to the
classifier
"""

import numpy as np
from ._architecture import RegionOfInterest, PeakResult, Peak
from typing import List

__all__ = ['defineRegions', 'filterPeakResults']


def filterPeakResults(peakresults: List[PeakResult], lower=40, upper=6000, snr=5) -> List[Peak]:
    """
    Filters peak results and returns a list of peaks

    Args:
        peakresults (list): List of PeakResult objects
        lower (optional, default 40): Lower limit
        upper (optional, default 6000): Upper limit
        snr (optional, default 5): Minimum snr to accept peaks

    Returns:
        List of filtered peak results
    """
    peaks = []
    for v in peakresults:
        pks = v.getPeaks()
        for pk in pks:
            # Chop out peaks outside the area of support
            if pk.energy < lower:
                continue
            if pk.energy > upper:
                continue
            if pk.intensity/np.sqrt(pk.baseline) < snr:
                continue
            peaks.append(pk)
            pk.uncovered = True
    peaks.sort(key=lambda p: p.energy)
    return peaks


def defineRegions(peakresults: List[PeakResult], sensor, limit=8, fraction=0.05, min_width=1, **kargs):
    """
    Heuristic algorithm that automatically selects the
    region of interest that describe a nuclide.

    This is used as part of the training process.

    Args:
        peakresults (list): List of PeakResult objects
        sensor: Sensor definition
        limit (optional, default 8): Maximum number of regions found.
        fraction (optional, default 0.15): Fraction definition the scale for looking at regions
        min_width (optional, default 1): Minimum width of the region in units of std. dev. at mean energy.

    Returns:
        Regions around the defined peaks
    """
    peaks = filterPeakResults(peakresults, **kargs)
    # Define the scales to search for regions
    n = len(peaks)
    scales = [int(4 * n * fraction), int(2 * n * fraction), int(n * fraction)]
    regions = []

    for scale in scales:
        while len(peaks) > scale:
            u = np.array([p.energy for p in peaks])

            # Look for a local grouping of peaks on the target scale
            u1 = u[scale:] - u[0:-scale]
            u2 = (u[scale:] + u[0:-scale]) / 2

            ix = np.argmin(u1)

            # Check to see if they are within one resolution of each other
            resolution = sensor.getResolution(u2[ix])
            if (u1[ix] > resolution / 2):
                # go to the next scale
                break

            roi = RegionOfInterest(
                u2[ix] - resolution / 2, u2[ix] + resolution / 2)
            p3 = [p.energy for p in peaks if p.energy in roi]
            mean = np.mean(p3)
            std = np.std(p3)
            # print(scale,resolution,6*std)
            width = np.max([3 * std, resolution * (0.5 * min_width)])
            roi = RegionOfInterest(mean - width, mean + width)
            peaks = [p for p in peaks if not p.energy in roi]
            regions.append(roi)

            if len(regions) == limit:
                break

    return regions


def computeCoverage(trainingSamples, regions):
    for ts in trainingSamples:
        ts.covered = False
        if len(ts.lines) == 0:
            ts.covered = True
            continue

        for e, i in ts.lines:
            if e < 50:
                continue

            # Skip regions we have already defined
            found = False
            for r in regions:
                if e in r:
                    found = True
                    break
            if (found):
                ts.covered = True
                continue


# FIXME Convert to bokeh!
'''
def plotCovered(peakresults, regions=None, **kargs):
    """ Plots the results of running the regions of interest file

    Args:
        peakresults:
        regions:
        **kargs:

    Returns:

    """
    plt.clf()
    plt.figure(figsize=(8, 6))
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    ax_scatter = plt.axes(rect_scatter)
    if regions:
        for r in regions:
            ax_scatter.axhspan(r.lower, r.upper, alpha=0.1)
    x = []
    y = []
    c = []
    for peakresult in peakresults:
        covered = False
        peaks = filterPeakResults([peakresult], **kargs)
        for pk in peaks:
            for r in regions:
                if pk.energy in r:
                    covered = True
                    break
            if covered:
                break

        for pk in peaks:
            x.append(peakresult.label.ad)
            y.append(pk.energy)
            if covered:
                c.append((0, 0, 0))
            else:
                c.append((1, 0, 0))
    x = np.array(x)
    y = np.array(y)

    ax_scatter.tick_params(direction='in', top=True, right=True)
    ax_scatter.scatter(x + 0.01, y, c=c, s=3)
    binwidth = 5
    lim = np.ceil(np.abs([x, y]).max() / binwidth) * binwidth * 1.1
    ax_scatter.set_ylim((0, lim))
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    bins = np.arange(0, lim + binwidth, binwidth)
    ax_histy.hist(y, bins=bins, orientation = 'horizontal')
    ax_histy.set_ylim(ax_scatter.get_ylim())
    ax_scatter.set_xlabel("AD (g/cm^2)")
    ax_scatter.set_ylabel("Energy (keV)")
    #gc = plt.gca()
    ax_scatter.set_xscale('log')
    ax_scatter.set_yscale('linear')
    ax_histy.set_yscale('linear')
    return ax_scatter, ax_histy
'''
