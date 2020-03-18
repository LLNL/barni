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
Feature extractor module.
"""

import numpy as np
import copy
from collections import OrderedDict
from typing import List
import pandas

from ._architecture import RegionOfInterest, FeatureExtractor, PeakResults, Features, Serializable
from ._reader import registerReader

__all__ = ["FeaturesROI", "FeatureExtractorNuclide"]


class FeatureExtractorNuclide(object):
    """
    (internal) representation for a nuclide.
    """

    def __init__(self):
        self.name = ""
        self._rois = []

    def setName(self, name: str):
        """
        Args:
            name: Name of the nuclide
        """
        self.name = name

    def addRegion(self, region: RegionOfInterest):
        """
        Args:
            region: Region of interest
        """
        self._rois.append(region)

    def addRegions(self, regions: List[RegionOfInterest]):
        for r in regions:
            self.addRegion(r)


class FeaturesROI(Features):
    def __init__(self):
        self._map = OrderedDict()

    def getFeatures(self) -> dict:
        return self._map

    def getDataFrame(self) -> pandas.DataFrame:
        return pandas.DataFrame(self.getFeatures(), index=[0])


class FeatureExtractorROI(FeatureExtractor, Serializable):
    '''
    Specific implementation of a feature extractor.
    '''

    def __init__(self):
        self._nuclides = []

    def extract(self, peaks: PeakResults) -> Features:
        out = FeaturesROI()
        for nuclide in self._nuclides:
            label = "Feature.%s." % nuclide.name
            counts = []
            for roi in nuclide._rois:
                result = peaks.sample.getRegionOfInterest(roi)
                c = result.intensity + np.sqrt(result.baseline)
                if peaks.intrinsic is not None:
                    result_intrinsic = peaks.intrinsic.getRegionOfInterest(roi)
                    c = c - result_intrinsic.intensity * peaks.scale_factor
                if c < 0:
                    c = 0
                counts.append(c)
            total = np.sum(counts)
            # out._map[label + "total"] = total
            if (total <= 0):
                total = 1
            for i, c in enumerate(counts):
                out._map[label + "roi%d" % i] = c / total
            out._map[label + "total"] = total
        return out

    def addNuclide(self, nuclide: FeatureExtractorNuclide):
        """ Adds a nuclide to the feature extractor
        Args:
            nuclide (FeatureExtractorNuclide):
        """
        self._nuclides.append(nuclide)

    def toXml(self):
        xml = "<FeatureExtractorROI>\n"
        for nuclide in self._nuclides:
            xml += "  <nuclide name='%s'>\n" % nuclide.name
            nuclide._rois.sort(key=lambda x: x.lower)
            for roi in nuclide._rois:
                xml += "    <roi lower='%.2f' upper='%.2f'/>\n" % (
                    roi.lower, roi.upper)
            xml += "  </nuclide>\n"
        xml += "</FeatureExtractorROI>\n"
        return xml

    def getTruthLabels(self):
        """ Create a set of labels to use for truth in a pandas dataframe """
        out = []
        for nuclide in self._nuclides:
            out.append(nuclide.name)
        return out

    def getFeatureLabels(self):
        """ Create a set of labels to use for features in a pandas dataframe """
        out = []
        for nuclide in self._nuclides:
            label = "Feature.%s." % nuclide.name
            for i, roi in enumerate(nuclide._rois):
                out.append(label + "roi%d" % i)
            out.append(label + "total")
        return out

    def strip_nuclide(self, nuclide: str):
        ''' Removes a nuclide from the list of extracted features. 

        In some cases the classifier could be retrained with fewer nuclides than provided by the extractor definition. This
        method removes the nuclide from the feature extractor. 

        Args:           
            nuclide: Nuclide to remove from the feature definitions. 

        Returns:
            new_features: Features with the nuclide removed.
        '''

        feature_extractor = copy.deepcopy(self)
        for fen in feature_extractor._nuclides:
            if fen.name == nuclide:
                feature_extractor._nuclides.remove(fen)
                break
        return feature_extractor

# Loaders


def loadRegionOfInterest(context, element):
    '''
    Converts a dom element into a RegionOfInterest object
    '''
    lower = float(element.attributes['lower'].value)
    upper = float(element.attributes['upper'].value)
    out = RegionOfInterest(lower, upper)
    return out


def loadFeatureExtractorNuclide(context, element):
    '''
    Converts a dom element into a Nuclide object
    '''
    out = FeatureExtractorNuclide()

    stringAttribs = ['name']

    for p, v in element.attributes.items():
        if p in stringAttribs:
            out.__setattr__(p, v)
            continue
        context.raiseAttributeError(element, p)

    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue

        if node.tagName == "roi":
            out._rois.append(loadRegionOfInterest(context, node))
            continue
        context.raiseElementError(element, node)

    return out


def loadFeatureExtractorROI(context, element):
    '''
    Converts a dom element into a FeatureExtractor object
    '''
    out = FeatureExtractorROI()

    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName == "nuclide":
            out._nuclides.append(loadFeatureExtractorNuclide(context, node))
            continue
        raise ValueError("Bad tag %s" % node.tagName)

    return out


registerReader("FeatureExtractorROI", loadFeatureExtractorROI)
# FIXME backward compatibility, remove in future release
registerReader("featureExtractor", loadFeatureExtractorROI)
