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
Classifiers for BARNI, the last step in the identification chain.

@author: monterial1
"""

from . import _architecture as arch
from ._result import NuclideResultList

from typing import List
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

__all__ = ['RandomForestClassifiers', 'RandomForestClassifications']


class RandomForestClassifications(arch.Classifications):
    """ Stores the results of the random forrest classification.
    """

    def __init__(self):
        self.nuclide_results = []

    def getNuclides(self) -> List[arch.NuclideResult]:
        """
        Get a list of nuclides that may be present in the sample.
        """
        return self.nuclide_results

    def addNuclideResult(self, result):
        self.nuclide_results.append(result)

    def getPredictions(self) -> NuclideResultList:
        """
        Returns: Nuclide results where prediction was made

        """
        nr_list = NuclideResultList()
        for nr in self.nuclide_results:
            if nr.prediction == 1:
                nr_list.addNuclideResult(nr)
        return nr_list


class RandomForestClassifiers(arch.Classifier):
    """ RandomForest implementation of the classifier.
    """

    def __init__(self):
        self._classifiers = {}
        self._thresholds = {}

    def add_classifier(self, cls: RandomForestClassifier, nuclide: str, thresh=0.5):
        """

        Args:
            cls: RandomForest model to add to the list.
            str: The name of the radionuclide that the model predicts
            thresh: Threashold for the classifier

        Returns:

        """
        self._classifiers[nuclide] = cls
        self._thresholds[nuclide] = thresh

    def train(self, features, truth, nuclide: str, **kargs):
        """ Traing a single classifier and add it to the list

        Args:
            features: Feature table used for training
            truth: Vector of truth, 1 for the nuclide of interest present in each row of the feature table.
            nuclide (str): Name of the nuclide associated with the classifier
        """

        rf = RandomForestClassifier(**kargs)
        cls = rf.fit(features, truth)
        cls.n_jobs = 1  # bug with predictions being slow due to thread-locking
        self.setThreshold(0.5, nuclide) # default threshold of 0.5
        self._classifiers[nuclide] = cls

    def train_all(self, features: DataFrame, truth: DataFrame, **kargs):
        """ Trains multiple classifiers at once, using the column labels from the training dable

        Args:
            features: Feature table as a pandas dataframe.
            truth: Truth trable as a pandas dataframe, the column labels will be nuclide names.
            n_estimators: Number of estimators (trees) to use for each classifier.
            max_depth: Maximum depths to use for each tree.
            n_jobs: The number of threads to spool to speed up training.
        """
        for nuclide in truth.columns:
            self.train(features, truth[nuclide], nuclide, **kargs)

    def getNuclides(self):
        """
        Returns: List of nuclides available for identification
        """
        return self._classifiers.keys()

    def setThreshold(self, threshold, nuclide):
        """ Set a threshold for a nuclide in the classifier
        Args:
            threshold: threshold between 0 and 1
            nuclide: Nuclide to assign the threshold

        """
        self._thresholds[nuclide] = threshold

    def classify(self, features: arch.Features) -> arch.Classifications:
        """ Perform classification from the set of input features

        Args:
            features: Set of input features

        Returns:
            Classification results
        """
        df = features.getDataFrame()
        classifications = RandomForestClassifications()
        for nuc, cls in self._classifiers.items():
            prob = cls.predict_proba(df)[0][1]
            predict = 1 if prob > self._thresholds[nuc] else 0
            nr = arch.NuclideResult(nuc, prob, predict)
            classifications.addNuclideResult(nr)
        return classifications
