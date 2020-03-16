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
Architecture for the entire BARNI package.
"""

from abc import ABC, abstractmethod
from typing import List
import pickle
import pathlib
import gzip
import textwrap

__all__ = [
    'IdentificationInput',
    'RegionOfInterest',
    'Peak',
    'PeakResult',
    'PeakAnalysis',
    'Features',
    'FeatureExtractor',
    'Classifications',
    'Classifier'
]

class Serializable(ABC):
    """ BARNI classes which can be serialized to XML or
        compressed XML classes.
    """

    @abstractmethod
    def toXml(self):
        """ Converts spectrum to XML string
        """

    def write(self, file : (pathlib.Path, str), compress=False):
        """ Dump template to XML file, or comptessed file
        Args:
            file (Path, str): Path to file.
            compress (default False): If true then apply compression
        """
        if compress:
            with gzip.open(file, "w") as fd:
                fd.write(bytes(self.toXml(),'utf-8'))
        else:
            with open(file, 'w') as fd:
                fd.write(self.toXml())

class IdentificationInput(Serializable):
    """
    Input to the identification object.

    This consists of a spectrum as a sample.  It may optionally have
    a background and a distance to sample (m).

    Attributes:
        sample (Spectrum): Sample spectrum.
        background (Spectrum, optional): Background spectrum.
        intrinsic (Spectrum, optional): Intrinsic source, typically used to perform continuous drift corrections.
        distance (float, optional, default 1): Source to detector distance.
        label (str, optional): Label for the input.
    """

    def __init__(self, sample, background=None, intrinsic=None, distance=1.0, label=None):
        self.sample = sample
        self.background = background
        self.intrinsic = intrinsic
        self.distance = distance
        self.label = label

    def toXml(self):
        """ Converts spectrum to XML string
        """
        xml = "<IdentificationInput>\n"
        xml += textwrap.indent(self.sample.toXml("sample"), "  ")
        if self.background is not None:
            xml += textwrap.indent(self.background.toXml("background"), "  ")
        if self.intrinsic is not None:
            xml += textwrap.indent(self.intrinsic.toXml("intrinsic"), "  ")
        if self.label is not None:
            xml += textwrap.indent(self.label.toXml(), "  ")
        xml += "  <distance>" + str(self.distance) + "</distance>\n"
        xml += "</IdentificationInput>\n"
        return xml

class RegionOfInterest(object):
    """  Generic concept of a region of interest.

    Attributes:
        lower (int): Lower bound of ROI.
        upper (int): Upper bound of ROI.
    """

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

    def __contains__(self, v):
        return v >= self.lower and v < self.upper

    def __str__(self):
        return "region(%f,%f)" % (self.lower, self.upper)


class Reader(ABC):
    """
    Handles reading in and conversion of different file types (e.g. N42, RTK)
    into BARNI types.
    """

    @abstractmethod
    def fromfile(self, file_path: pathlib.Path):
        """

        Args:
            file_path (Path): Full path to the data object.
        """

##########################################################################

# The peak analysis processes the raw sample spectrum into arbitrary number of
# spectral features that describes the spectrum.  Because this produces as many
# features as needed to describe the input sample, the features are processed
# into a uniform set prior to passing to the classifier.

# the input to a peak extractor is a IdentificationInput


class Peak(Serializable):
    """ Representation of an individual peak extracted by the PeakAnalysis.

    Attributes:
        energy (float): The extracted location of the peak.
        intensity (float): The total number of integrated counts in the peak.
            This is measured across the peak width (typically +/- a FWHM)
        baseline (float): The total amount of baseline under the peak.
            This is measured on the same interval as the peak.
    """

    def __init__(self, energy, intensity, baseline, width=0):
        self.energy = energy
        self.intensity = intensity
        self.baseline = baseline
        self.width = width

    def __str__(self):
        return "energy %f, intensity %f, width %f" % (self.energy, self.intensity, self.width)

    def toXml(self):
        """ Converts spectrum to XML string

        """

        xml = "<Peak>\n"
        attributes = ["energy", "intensity", "baseline", "width"]
        for attr in attributes:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if value is not None:
                    xml += "  <%s>" % attr
                    xml += str(value)
                    xml += "</%s>\n" % attr
        xml += "</Peak>\n"
        return xml



class PeakResult(Serializable):
    """ Results from the peak extraction procedure.
    """

    @abstractmethod
    def getPeaks(self) -> List[Peak]:
        """
        Get the list of peaks that were extracted.
        """

    @abstractmethod
    def getRegionOfInterest(self, roi: RegionOfInterest) -> Peak:
        """
        Get the sum of all the peaks in the region of interest
        and the statistics for the baseline.
        """

    @abstractmethod
    def getContinuum(self):
        """ Gets the estimate of the continuum

        Returns: Spectrum
        """

    @abstractmethod
    def getFit(self):
        """ This is the sum of continuum and responsed peaks
        Returns: Spectrum
        """

class PeakResults(Serializable):
    """ Container for peak results for different components of identification input.

    Attributes:
        sample (PeakResult): The peak results from the sample.
        intrinsic (PeakResult, optional): The peak result from the instrinsic source.
        scale_factor (float, optional): Scaling factor used to normalize the intrinsic source to
            the sample, used during subtraction in the feature extraction process.
    """

    def __init__(self, sample: PeakResult, intrinsic: PeakResult = None, scale_factor: float = None):
        self.sample = sample
        self.intrinsic = intrinsic
        self.scale_factor = scale_factor

    def toXml(self):
        xml = "<PeakResults>\n"
        xml += textwrap.indent(self.sample.toXml("sample"), "  ")
        if self.intrinsic is not None:
            xml += textwrap.indent(self.intrinsic.toXml("intrinsic"), "  ")
        if self.scale_factor is not None:
            xml += "  <scale_factor>"
            xml += str(self.scale_factor)
            xml += "</scale_factor>\n"
        xml += "</PeakResults>\n"
        return xml

class PeakAnalysis(ABC):
    """
    Algorithm which takes the features produced by the feature extractor and
    converts it into an identification of nuclides.
    """

    @abstractmethod
    def analyze(self, sample):
        """
        Args:
            sample (IdentificationInput): Identification input to find peaks in.

        Returns:
            PeakResults object.
        """


##########################################################################

# The feature extractor takes the result of the peak analysis and extracts a
# uniform set of features to present the classifier.  Typically it uses regions
# of interest for individual nuclides which are matched against the results of the
# peak analysis.  These regions are defined either by an expert or by a
# machine process.


class Features(ABC):
    """
    Result from a classifier holding a list of nuclides with the probability of
    presence in the sample.

    This will be nothing more than a map holding a feature name and a float
    with presence of a source.
    """

    def getFeatures(self) -> dict:
        """ Returns the map of strings and float values. """

    def getDataFrame(self):
        """ Returns data frame representation of the features """

class FeatureExtractor(ABC):
    """
    Algorithm which takes the features produced by the feature extractor and
    converts it into an identification of nuclides.
    """

    @abstractmethod
    def extract(self, peaks: PeakResults) -> Features:
        """
        Computes the classifier results from the set of features
        """

    @abstractmethod
    def getTruthLabels(self):
        """ Get the truth labels """

    @abstractmethod
    def getFeatureLabels(self):
        """ Get the feature labels """

##########################################################################

# The classifier takes a list of uniform features defined by the feature extractor
# and applies a seperate classifier for each nuclide in our set of interest.
# It then produces a result indicating the presence of sources that it
# analyzed.

class NuclideResult(Serializable):
    """ An individual nuclide that was determined to be present.

    This will have a name of the nuclide and a probabiliy of presence.
    It may optionally hold additional information that was associated
    with this nuclide.

    Attributes:
        nuclide (str): Name of the nuclide.
        score (float): score of nuclide's presence in the sample.
        prediction (int): 1 if nuclide found otherwise 0.
    """

    def __init__(self, nuclide: str, score: float, prediction: int):
        self.nuclide = nuclide
        self.score = score
        self.prediction = prediction

    def __str__(self):
        return "%s , %f" % (self.nuclide, self.score)

    def toXml(self):
        """ Converts spectrum to XML string

        """
        xml = "<NuclideResult>\n"
        attributes = ["nuclide", "score", "prediction"]
        for attr in attributes:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if value is not None:
                    xml += "  <%s>" % attr
                    xml += str(value)
                    xml += "</%s>\n" % attr
        xml += "</NuclideResult>\n"
        return xml

class Classifications(ABC):
    """
    Result from a classifier holding a list of nuclides with the probability of
    presence in the sample.
    """

    @abstractmethod
    def getNuclides(self) -> List[NuclideResult]:
        """
        Get a list of nuclides that may be present in the sample.
        """

    @abstractmethod
    def getPredictions(self):
        """ Gets a NuclideResultsList where prediction is equal to 1.
        """


class Classifier(ABC):
    """
    Algorithm which takes the features produced by the feature extractor and
    converts it into an identification of nuclides.
    """

    @abstractmethod
    def classify(self, features: Features) -> Classifications:
        """
        Computes the classifier results from the set of features
        """

    def save(self, filename):
        """ Saves the classifier to a pickle file.

        Args:
           filename (str, or Path): Name of the pickle file.
        """
        f = open(filename, 'wb')
        pickle.dump(self, f)
        f.close()

    @classmethod
    def load(cls, filename):
        """ Loads in the classifier from a pickled file.

        Args:
           filename: Filename of the pickle holding the classifier.

        Returns:
            Instance of a classifer class saved.
        """
        return pickle.load(open(filename, "rb"))


##########################################################################


class SensorModel(Serializable):
    """
    Sensor model calculates the response of a detector to incident flux.
    """

    @abstractmethod
    def getResponse(self, energy, intensity, binEdges):
        """ Integral of gaussian of intensity between two bins """

    @abstractmethod
    def getResponseIntegral(self, energy1, energy2,
                            intensity1, intensity2, binEdges):
        """ Evaluate the response integral, uses Simpson's Rule """

    @abstractmethod
    def getResolution(self, energy):
        """ Evaluate the detector resolution at an energy """


class TrainingStageInput():
    """
    Base class of inputs for each input stage.
    """

    def unpack(self, doc):
        self.save = bool(doc["save"])
        self.skip = bool(doc["skip"])

    def SetSkip(self, skip : bool):
        """
        Args:
            skip: If true the stage is skipped over.
        """
        self.skip = skip

    def SetSave(self, save : bool):
        """
        Args:
            save (bool) : If true the output of the stage are saved.
                This output is required for
        """
        self.save = save

