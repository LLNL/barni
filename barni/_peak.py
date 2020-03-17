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
Module for handling peak definitions

@author monterial1
"""

import textwrap
from collections import UserList

from ._architecture import Peak, PeakResults, Serializable
from ._spa import SmoothPeakResult
from ._spectrum import loadSpectrum
from ._sensor import loadGaussianSensorModel
from ._reader import registerReader

__all__ = ["PeakResultsList"]

def loadPeak(context, element):
    """ Loads in a Peak
    """

    floatFields = [
        'energy',
        'intensity',
        'baseline',
        'width'
    ]

    out = Peak(energy = None, intensity = None, baseline = None, width=0)
    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName in floatFields:
            out.__setattr__(node.tagName, float(node.firstChild.nodeValue))
            continue
        context.raiseElementError(element, node)
    return out

def loadSmoothPeakResult(context, element):
    """ Load in a smooth peak result
    """

    out = SmoothPeakResult(peaks = [], continuum=None, sensor=None)
    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName == "Peak":
            out._peaks.append(loadPeak(context, node))
            continue
        if node.tagName == "Spectrum" and node.attributes['name'].value == "continuum":
            out._continuum = loadSpectrum(context, node)
            continue
        # FIXME Every sensor model will need to be added manually this way
        if node.tagName == "GaussianSensorModel":
            out._sensor = loadGaussianSensorModel(context, node)
            continue
        context.raiseElementError(element, node)
    return out

def loadPeakResults(context, element):
    """
    Reads in spectrum from XML document.
    """

    out = PeakResults(sample = None)
    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if  node.tagName == "SmoothPeakResult":
            if node.attributes['name'].value == "sample":
                out.sample = loadSmoothPeakResult(context, node)
                continue
            if node.attributes['name'].value == "intrinsic":
                out.intrinsic = loadSmoothPeakResult(context, node)
                continue
        if node.tagName == "scale_factor":
            out.scale_factor = float(node.firstChild.nodeValue)
            continue
        context.raiseElementError(element, node)
    return out


class PeakResultsList(Serializable, UserList):
    """ List of PeakResults

    Useful class for holding the intermediate results of the training routine.
    """

    def addPeakResults(self, input : PeakResults):
        self.data.append(input)

    def toXml(self):
        xml = "<PeakResultsList>\n"
        for pr in self.data:
            xml += textwrap.indent(pr.toXml(), "  ")
        xml += "</PeakResultsList>\n"
        return xml


def loadPeakResultsList(context, element):
    out = PeakResultsList()
    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName == "PeakResults":
            out.addPeakResults(loadPeakResults(context, node))
            continue
        context.raiseElementError(element, node)
    return out


registerReader("PeakResults", loadPeakResults)
registerReader("PeakResultsList", loadPeakResultsList)