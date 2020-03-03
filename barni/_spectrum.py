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
Module for spectrum and templates.
"""

import numpy as np
import copy
from scipy import spatial
import textwrap
from collections import UserList

from ._architecture import Serializable
from ._bins import EnergyScale, loadEnergyScale
from ._label import Label, loadLabel
from ._reader import registerReader

__all__ = ["Spectrum", "Template", "draw_spectrum", "filter_templates", "TemplateList", "SpectrumList"]


class Spectrum(Serializable):
    """
    Class representing a histogram typically recorded
    by a gamma sensor.

    Every spectra is guaranteed to have
    an array of counts and a corresponding energy scale.
    There are a list of other fields that can be populated
    if the information is available.

    Attributes:
        counts (int): Number of total counts in the spectrum
        energyScale (EnergyScale): The energy scale of the spectrum.
        rt (float): Real time in seconds.
        lt (float): Live time in seconds.
        dose (float): Dose rate at the detector face in units (uR).
        distance (float): Distance from source in cm.
        title (str): String associated with the spectra.
    """

    def __init__(self, counts, energyScale, rt=1, lt=1, distance = None, gamma_dose = None, title = None):
        self.counts = np.array(counts)
        self.energyScale = energyScale
        self.livetime = lt
        self.realtime = rt
        self.distance = distance
        self.gamma_dose = gamma_dose
        self.title = title

    def getIntegral(self, e1, e2):
        # Find the starting and ending points in the energy scale
        c3 = self.energyScale.findBin(e1)
        c4 = self.energyScale.findBin(e2)
        energyBins = self.energyScale.getEdges()
        # Compute the partial bins
        u1 = energyBins[c3]
        u2 = energyBins[c3 + 1]
        f1 = (e1 - u1) / (u2 - u1)

        u1 = energyBins[c4]
        u2 = energyBins[c4 + 1]
        f2 = (e2 - u1) / (u2 - u1)

        # Total is the sum of the whole bins minus the two partials
        return np.sum(self.counts[c3:c4 + 1]) - \
            self.counts[c3] * f1 - self.counts[c4] * f2

    def downsample(self):
        """ Downsample a spectrum by adding up every other bin.

        FIXME this should likely be a utility function rather than a member.
        """
        counts = self.counts[0::2] + self.counts[1::2]
        rt = self.realtime
        lt = self.livetime
        es = EnergyScale(self.energyScale[range(0, len(self.energyScale), 2)])
        out = Spectrum(counts, es, rt, lt)

        # Copy attributes
        for attr in ["label", "title"]:
            if hasattr(self, attr):
                setattr(out, attr, getattr(self, attr))
        return out

    def copy(self):
        return copy.deepcopy(self)

    def getNormedCounts(self):
        """
        Calculates counts normalized by the bin width.

        Returns:
            counts (array): Counts per energy
        """
        edges = np.array(self.energyScale.getEdges())
        bin_width = edges[1:] - edges[:-1]
        counts = self.counts / bin_width
        return counts

    def toXml(self, name = None):
        """ Converts spectrum to XML string

        Args:
            name (str): Attribute of the spectrum, useful for different spectra names
        """

        if name is None:
            xml = "<Spectrum>\n"
        else:
            xml = "<Spectrum name='%s'>\n" % name
        xml += "  <counts>"
        for count in self.counts:
            xml += str(count) + " "
        xml += "</counts>\n"
        xml += textwrap.indent(self.energyScale.toXml(), "  ")
        attributes = ["livetime", "realtime", "distance", "gamma_dose", "title"]
        for attr in attributes:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if value:
                    xml += "  <%s>" % attr
                    xml += str(value)
                    xml += "</%s>\n" % attr
        xml += "</Spectrum>\n"
        return xml

class SpectrumList(Serializable, UserList):
    """ List of Spectra
    """
    def addSpectrum(self, input : Spectrum):
        self.data.append(input)

    def toXml(self):
        xml = "<SpectrumList>\n"
        for tmp in self.data:
            xml += textwrap.indent(tmp.toXml(), "  ")
        xml += "</SpectrumList>\n"
        return xml


def draw_spectrum(spectrum : Spectrum, counts):
    ''' Re-samples counts from a provided spectra using poisson statistics.

    Ideally the provided spectra should be a normalized model.

    Args:
        spectrum (Spectrum): Input spectrum.
        counts (int): Number of counts to draw.

    Returns:
        Spectrum with drawn number of counts.
    '''

    s3 = copy.deepcopy(spectrum)
    s3.counts *= float(counts) / float(np.sum(s3.counts))
    s3.counts = np.random.poisson(s3.counts)
    return s3


class Template(Serializable):
    """
    Container templates used to generate input samples for BARNI.

    Attributes:
        spectrum (Spectrum): The template spectrum.
        label (Label): Label containing additional information on the spectrum.
    """

    def __init__(self, spectrum: Spectrum, label: Label):
        self.spectrum = spectrum
        self.label = label

    def toXml(self):
        """ Converts spectrum to XML string
        """

        xml = "<Template>\n"
        xml += textwrap.indent(self.spectrum.toXml(), "  ")
        xml += textwrap.indent(self.label.toXml(), "  ")
        xml += "</Template>\n"
        return xml

class TemplateList(Serializable, UserList):
    """ List of Templates

    Useful class for holding the intermediate results of the training routine.
    """

    def addTemplate(self, input : Template):
        self.data.append(input)


    def toXml(self):
        xml = "<TemplateList>\n"
        for tmp in self.data:
            xml += textwrap.indent(tmp.toXml(), "  ")
        xml += "</TemplateList>\n"
        return xml


def filter_templates(templates, z=0, ad=0):
    """ Filters templates by desired shielding configuration

    Uses a k-nearest-neighbor search to find the combination
    of aerial densitry and atomic number in the templates
    that is closes to the desired configuration.

    Args:
        templates (list): List of templates
        z: The desired atomic number. 
        ad: The desired aerial density.

    Returns:
        List of templates that closest to the desired shielding configuration.
    """

    shielding = []
    for template in templates:
        label = template.label
        shielding.append([label.z, label.ad])
    shielding = np.array(shielding)
    ind = int(spatial.KDTree(shielding).query([z, ad])[1])
    return templates[ind]

def loadSpectrum(context, element):
    """
    Reads in spectrum from XML document.
    """

    floatFields = [
        'realtime',
        'livetime',
        'distance',
        'gamma_dose'
    ]

    stringFields = ['title']

    out = Spectrum(counts = None, energyScale = None)
    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName == "counts":
            counts = np.fromstring(node.firstChild.data, sep=" ")
            continue
        if node.tagName == "EnergyScale":
            energyScale = loadEnergyScale(context, node)
            continue
        if node.tagName in floatFields:
            out.__setattr__(node.tagName, float(node.firstChild.nodeValue))
            continue
        if node.tagName in stringFields:
            out.__setattr__(node.tagName, str(node.firstChild.nodeValue))
            continue
        context.raiseElementError(element, node)
    out.counts = counts
    out.energyScale = energyScale
    return out

def loadTemplate(context, element):
    """
    Reads in template
    """

    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName == "Label":
            label = loadLabel(context, node)
            continue
        if node.tagName == "Spectrum":
            spectrum = loadSpectrum(context, node)
            continue
        context.raiseElementError(element, node)
    template = Template(spectrum, label)
    return template


def loadTemplateList(context, element):
    out = TemplateList()
    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName == "Template":
            out.addTemplate(loadTemplate(context, node))
            continue
        context.raiseElementError(element, node)
    return out

def loadSpectrumList(context, element):
    out = SpectrumList()
    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName == "Spectrum":
            out.addSpectrum(loadSpectrum(context, node))
            continue
        context.raiseElementError(element, node)
    return out

registerReader("Spectrum", loadSpectrum)
registerReader("SpectrumList", loadSpectrumList)
registerReader("Template", loadTemplate)
registerReader("TemplateList", loadTemplateList)