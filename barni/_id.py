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
Identification module.
"""

import yaml
from typing import List
import textwrap
from collections import UserList

from ._reader import loadXml, registerReader
from ._architecture import FeatureExtractor, Classifier, PeakAnalysis, Serializable, IdentificationInput
from ._spectrum import loadSpectrum
from ._label import loadLabel

__all__ = [
    'IdentificationAlgorithm',
    'IdentificationResult',
    'IdentificationInputList'
]



class IdentificationInputList(Serializable, UserList):
    """ Holds a list of Identification Inputs.

        This is a convinience class used for training,
        but it also encapsulates the list of identification
        inputs when the results are cached.
    """

    def addInput(self, input : IdentificationInput):
        self.data.append(input)

    def toXml(self):
        xml = "<IdentificationInputList>\n"
        for input in self.data:
            xml += textwrap.indent(input.toXml(), "  ")
        xml += "</IdentificationInputList>\n"
        return xml

class IdentificationResult(object):
    """
    The final result of the identification algorithm.

    Attributes:
      input (barni.IdentificationInput): The initial sample provided to the algorithm.
      peaks (PeakResults):  The peaks extracted from the spectrum (along with any other data extracted
        from the raw spectrum.
      features (Features): The set of features computed from the extracted peaks.
      classifications (Classifications): The set of nuclides that were determined to be present based
        on the features.
    """

    def __init__(self):
        self.input = None
        self.peaks = None
        self.features = None
        self.classifications = None


class IdentificationAlgorithm(object):
    """
    This is the main algorithm.
    To use load the algorithm with a configuration.  Then request an identification
    to produce an identification result.

    Attributes:
        peakExtractor (PeakAnalysis): Peak finding algorithm which produces peak results.
        featureExtraction (FeatureExtractor): Feature extraction algorithm.
        classifier (Classifier): Takes the input of features and produces classifications.

    """

    def __init__(self):
        self.peakExtractor = None
        self.featureExtractor = None
        self.classifier = None

    def setPeakExtractor(self, peakExtractor):
        self.peakExtractor = peakExtractor

    def setFeatureExtractor(self, featureExtractor: FeatureExtractor):
        self.featureExtractor = featureExtractor

    def setClassifier(self, classifier: Classifier):
        self.classifier = classifier

    def identify(self, id_input: IdentificationInput) -> IdentificationResult:
        result = IdentificationResult()
        result.input = id_input
        result.peaks = self.peakExtractor.analyze(result.input)
        result.features = self.featureExtractor.extract(result.peaks)
        result.classifications = self.classifier.classify(result.features)
        return result

class IdentificationAlgorithmInput(object):
    """
    The inputs required for the identification algorithm for the command line interface.
    """

    def __init__(self, peakanalysis : PeakAnalysis = None, feature_extractor : FeatureExtractor = None,
                       classifiers : Classifier = None):
        self.peakanalysis = peakanalysis
        self.feature_extractor = feature_extractor
        self.classifiers = classifiers

    def fromfile(self, filename):
        """ Loads in the YAML file with training input options

        Args:
            filename: Name of the yaml file to load.
        """
        with open(filename, "r") as file:
            doc = yaml.load(file, Loader=yaml.FullLoader)
        self.unpack(doc)

    def unpack(self, doc):
        """ Loads in input requirements from a dictionary

        Args:
            doc: Dictionary holding inputs for all stages of BARNI training.
        """

        self.peakanalysis = loadXml(doc["peakanalysis"])
        self.feature_extractor = loadXml(doc["feature_extractor"])
        self.classifiers = Classifier.load(doc["classifier"])

def loadIdentificationInput(context, element):
    """
    Reads in spectrum from XML document.
    """

    out = IdentificationInput(sample = None)
    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName == "Spectrum" and node.attributes['name'].value == "sample":
            out.sample = loadSpectrum(context, node)
            continue
        if node.tagName == "Spectrum" and node.attributes['name'].value == "background":
            out.background = loadSpectrum(context, node)
            continue
        if node.tagName == "Spectrum" and node.attributes['name'].value == "intrinsic":
            out.intrinsic = loadSpectrum(context, node)
            continue
        if node.tagName == "distance":
            out.distance = float(node.firstChild.nodeValue)
            continue
        if node.tagName == "label":
            out.label = loadLabel(context, node)
            continue
        context.raiseElementError(element, node)
    return out


def loadIdentificationInputList(context, element):
    """
    Reads in spectrum from XML document.
    """

    out = IdentificationInputList()
    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName == "IdentificationInput":
            out.addInput(loadIdentificationInput(context, node))
            continue
        context.raiseElementError(element, node)
    return out

registerReader("IdentificationInput", loadIdentificationInput)
registerReader("IdentificationInputList", loadIdentificationInputList)
