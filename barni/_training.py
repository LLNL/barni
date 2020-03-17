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
Training module.

@author monterial1
"""

import numpy as np
from pathlib import Path
import yaml
import pandas
import os
from random import choices
from ._architecture import FeatureExtractor, PeakAnalysis, TrainingStageInput, IdentificationInput
from ._id import IdentificationInputList
from ._peak import PeakResultsList
from ._spectrum import  draw_spectrum, Template, TemplateList
from ._roi import defineRegions
from ._fe import FeatureExtractorROI, FeatureExtractorNuclide
from ._class import RandomForestClassifiers
from ._reader import loadXml

__all__ = ["TrainProcessor", "SampleGenerator", "PeakProcessor", "TrainingInput", "FeatureBuilder"]


class SampleGenerator(object):
    """ Generates IdenfificiatioInputs from provided templates.

    Attributes:
        templates: List of templates to draw from.

    """


    def getInputs(self, templates_list : TemplateList, counts,
                  samples : int = None, background : Template = None, fraction = None) -> IdentificationInputList:
        """
        Args:
            templates_list: TemplateList to draw from for sampling.
            counts: Either a number of counts to sample
                spectra from or a function which will generate
                the counts when called.
            samples (int, default None): Number of samples to draw (with replacement).
                By default one sample is drawn per template.
            bacgkround: Template to use for drawing the background distribution.
            fraction: Function producing the fraction of background to add to the sample.

        Returns:
            id_inptus (IdentificationInputList): List of identification inputs.
        """

        if hasattr(counts, "__float__"):
            v = counts
            def counts(): return v

        if hasattr(fraction, "__float__"):
            u = fraction
            def fraction(): return u

        if samples is None:
            templates = templates_list
        else:
            templates = choices(templates_list, k=samples)
        id_inputs = IdentificationInputList()
        for template in templates:
            sample_counts = float(counts())
            sample_spectrum = draw_spectrum(template.spectrum, sample_counts)
            if background and fraction:
                bkg_counts = sample_spectrum.counts.sum() * fraction()
                bkg_spectrum = draw_spectrum(background.spectrum, bkg_counts)
                sample_spectrum.counts = sample_spectrum.counts + bkg_spectrum.counts
            id_input = IdentificationInput(sample_spectrum, label=template.label)
            id_inputs.addInput(id_input)
        return id_inputs


class PeakProcessor(object):
    """
    Processes list of identification inputs and returns a list of peak results.

    Attributes:
        spa (PeakAnalysis): Peak analysis algorithm to use for processing.
    """

    def __init__(self, spa : PeakAnalysis):
        self.spa = spa

    def analyze(self, id_inputs : IdentificationInputList) -> PeakResultsList:
        """ Runs the peak analysis on provided list of inputs

        Args:
            id_inputs (IdentificationInputList): List of identificaiton inputs to process

        Returns:
            peak_results (PeakResultList): List of peak results.

        """

        results = PeakResultsList()
        for entry in id_inputs:
            result = self.spa.analyze(entry)
            result.sample.label = entry.label
            results.addPeakResults(result)
        return results


class FeatureBuilder(object):
    """
    Processes a list of peak results into features and creates a feature
    and truth tables required for training

    Attributes:
        feature_extractor (FeatureExtractor): Feature extractor
    """

    def __init__(self, feature_extractor : FeatureExtractor):
        self.feature_extractor = feature_extractor

    def build(self, peakresults : PeakResultsList, nuclide : str):
        """
        Processes all the peak results from a specified nuclide and creates a feature and truth table.

        Args:
            peakresults: List of peak results.
            nuclide: The nuclide present in each peak result.

        Returns:
            feature_table: Table of all the features.
            truth_table: The corresponding truth table.
        """

        truthLabels = self.feature_extractor.getTruthLabels()
        featureLabels = self.feature_extractor.getFeatureLabels()
        label = {nuclide: 1}
        truth0 = pandas.DataFrame(
            0, columns=truthLabels, index=range(
                0, len(peakresults)), dtype=np.int)
        features0 = pandas.DataFrame(
            0, columns=featureLabels, index=range(
                0, len(peakresults)), dtype=np.float)
        if (nuclide!="bkg" and nuclide!="background"):
            truth0.loc[:, label] = 1
        for i, sample in enumerate(peakresults):
            f = self.feature_extractor.extract(sample)
            features0.iloc[i] = f.getDataFrame().iloc[0]
        return features0, truth0


class SampleGeneratorInput(TrainingStageInput):
    """
    Unpacks input generation input requirements.

    Attributes:
        template (TemplateInput): Template input holder class.
        counts (function): Returns a range of counts for the sample drawing.
        samples (int): Number of samples to draw from templates with replacement
        background (Template): Background template used for sampling. The reader returns a list,
            only the first template is grabbed from that list.
        fraction (function): Returns the fraction of background contribution in the drawn sample.
        output (str) : The name of the output file.
    """
    def __init__(self, template = None, counts = None, samples = None, background = None, fraction = None, output = None):
        if template is None:
            self.template = self.TemplateInput()
        else:
            self.template = template
        self.samples = samples
        self.counts = counts
        self.background = background
        self.fraction = fraction
        self.output = output

    def unpack(self, doc):
        """ Unpacks dictionary of input options
        """
        super().unpack(doc)
        self.output = doc["output"]
        self.template.unpack(doc["template"])
        self.counts = lambda : np.exp(np.random.uniform(np.log(doc['counts_low']), np.log(doc['counts_high'])))
        if type(doc["samples"]) is int:
            self.samples = int(doc["samples"])
        elif str(doc["samples"]).lower() == "none":
            self.samples = None
        else:
            raise RuntimeError("Sample generator paremeter samples must be either an integer or 'None'")
        if "background" in doc:
            self._unpack_background(doc["background"])
        else:
            self.background = None
            self.fraction = None

    def _unpack_background(self, doc):
        # eval is quite dangerous on untested strings
        filepath = Path(doc["filepath"])
        self.background = loadXml(str(filepath))
        self.fraction = lambda : np.random.uniform(doc["fraction_low"], doc["fraction_high"])

    class TemplateInput():
        """ Holds inputs for template reading

        See TemplateReader class attributes.

        Attributes:
            sourcepath: The template directory with all the nuclide folders.
            filename: The name of file with TemplateList inside the nuclide folders.
        """

        def __init__(self, sourcepath = None, filename = None):
            self.sourcepath = sourcepath
            self.filename = filename

        def unpack(self, doc):
            self.sourcepath = Path(doc['directory'])
            self.filename = str(doc['filename'])


class PeakAnalysisInput(TrainingStageInput):
    """
    Unpacks and holds peak generation inputs

    Attributes:
        sample_generator (SampleGeneratorInput): The input for generating samples.
        spa (PeakAnalysis): Peak analysis method to use for finding peaks.
        output(str) : Output file name.
    """

    def __init__(self, sample_generator : SampleGeneratorInput = None, spa : PeakAnalysis = None, output : str = None):
        if sample_generator is None:
            self.sample_generator = SampleGeneratorInput()
        else:
            self.sample_generator = sample_generator
        self.spa = spa
        self.output = output

    def unpack(self, doc):
        super().unpack(doc)
        self.sample_generator.unpack(doc["sample_generator"])
        self.output = str(doc["output"])
        self.spa =  loadXml(doc["peakfile"])


class RoiDefinerInput(TrainingStageInput):
    """
    Unpacks and holds roi definer inputs

    Attributes:
        peakanalysis (PeakAnalysisInput): The input for the peak finding routine.
        counts (int): Number of counts to sample for the training routine
        snr (float): Minimum snr for peaks used in defining roi.
        limit (int): Maximum number of roi per nuclide.
        output (str): ROI output file name holder.

    """

    def __init__(self, peakanalysis=None, snr = None, limit = None, min_width = None, output = None):

        if peakanalysis is None:
            self.peakanalysis = PeakAnalysisInput()
        else:
            self.peakanalysis = peakanalysis
        self.snr = snr
        self.limit = limit
        self.min_width = min_width
        self.output = output

    def unpack(self, doc):
        super().unpack(doc)
        self.peakanalysis.unpack(doc["peakanalysis"])
        self.output = str(doc["output"])
        self.snr = float(doc["snr"])
        self.limit = int(doc["limit"])
        self.min_width = int(doc["min_width"])

class FeatureBuilderInput(TrainingStageInput):
    """
    Unpacks the feature and truth table definition inputs.

    Attributes:
        roi_definer: Roi definer input
        output : Table builder output files
    """

    def __init__(self, peakanalysis : PeakAnalysisInput = None, roi_definer : RoiDefinerInput = None, output = None):
        if peakanalysis is None:
            self.peakanalysis = PeakAnalysisInput()
        else:
            self.peakanalysis = peakanalysis
        if roi_definer is None:
            self.roi_definer = RoiDefinerInput()
        else:
            self.roi_definer = roi_definer
        if output is None:
            self.output = self.OutputFiles()
        else:
            self.output = output

    def unpack(self, doc):
        super().unpack(doc)
        self.peakanalysis.unpack(doc["peakanalysis"])
        self.roi_definer.unpack(doc["roi_definer"])
        self.output.unpack(doc["output"])

    class OutputFiles():
        def __init__(self, features=None, truth=None):
            self.features = features
            self.truth = truth

        def unpack(self, doc):
            self.features = str(doc["features"])
            self.truth = str(doc["truth"])


class ClassifierBuilderInput(TrainingStageInput):
    """ Inputs for the classifier (random forest only)
    """
    def __init__(self,  feature_builder = None, n_estimators = None, max_depth = None, n_jobs = None, output = None):
        if feature_builder is None:
            self.feature_builder = FeatureBuilderInput()
        else:
            self.feature_builder = feature_builder
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.output = output

    def unpack(self, doc):
        super().unpack(doc)
        self.feature_builder.unpack(doc["feature_builder"])
        self.output = str(doc["output"])
        self.n_estimators = int(doc["n_estimators"])
        self.max_depth = self._read_int(doc["max_depth"], "max_depth")
        self.n_jobs = self._read_int(doc["n_jobs"], "n_jobs")

    def _read_int(self, doc, name):
        """ Reads integer parameters
        """
        if type(doc) is int:
            para = int(doc)
        elif str(doc).lower() == "none":
            para = None
        else:
            raise RuntimeError("Classifier paremeter %s must be either an integer or 'None'" % name)
        return para


class TrainingInput():
    """
    Holds all the stages of the training routine.

    Attributes:
        classifier_builder: Classifier inputs.
        buildpath: Path for building files into
        nuclides: The number of nuclides to perform training on.
    """

    def __init__(self):
        self.classifier_builder = ClassifierBuilderInput()
        self.buildpath = None
        self.nuclides = None

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
        self.classifier_builder.unpack(doc["classifier_builder"])
        self.buildpath = Path(doc["buildpath"])
        self.nuclides = doc["nuclides"]


class TrainProcessor():
    """
    Runs through and performs the training requrested by the input
    """

    def __init__(self, train_input):
        self.train_input = train_input

    def train(self):
        """
        Args:
            train_input: The setup inputs for all training stages.
        """

        # Classifier builder step
        if self.train_input.classifier_builder.skip:
            # this call is to ensure that feature extractor building can be called even if classification is skipped
            self.getFeatureExtractor(self.train_input.classifier_builder.feature_builder.roi_definer)
            print("Skipping the classifier building")
        else:
            features, truth = self.getFeatures(self.train_input.classifier_builder.feature_builder)
            classifiers = RandomForestClassifiers()
            n_estimators = self.train_input.classifier_builder.n_estimators
            max_depth = self.train_input.classifier_builder.max_depth
            n_jobs = self.train_input.classifier_builder.n_jobs
            for nuclide in self.train_input.nuclides:
                # skip training for background
                if nuclide == "bkg" or nuclide == "background":
                    print("Skipping classifier building for %s"  % nuclide)
                    continue
                print("training classifier for %s \n" % nuclide)
                classifiers.train(features, truth[nuclide], nuclide, n_estimators=n_estimators, max_depth=max_depth, n_jobs = n_jobs)
            if self.train_input.classifier_builder.save:
                cls_file = self.train_input.buildpath.joinpath(self.train_input.classifier_builder.output)
                os.makedirs(cls_file.parent, exist_ok=True)
                classifiers.save(cls_file)


    def getFeatures(self, feature_builder_input : FeatureBuilderInput):
        """ Retrieves the feature and truth data frames
        Args:
            feature_builder: Feature builder input definitions
        """
        f_file = self.train_input.buildpath.joinpath(feature_builder_input.output.features)
        t_file = self.train_input.buildpath.joinpath(feature_builder_input.output.truth)
        feature_extractor = self.getFeatureExtractor(feature_builder_input.roi_definer)
        if feature_builder_input.skip:
            if f_file.exists() and t_file.exists():
                print("Skipping feature building, reading from %s and %s" % (f_file.name, t_file.name))
                features = pandas.read_csv(f_file)
                truth = pandas.read_csv(t_file)
            else:
                raise RuntimeError("%s or %s files not found" % (f_file.name, t_file.name))
        else:
            # check that the feature extractor has all the nuclides
            fe_nuclides = feature_extractor.getTruthLabels()
            if set(fe_nuclides) != set(self.train_input.nuclides):
                raise RuntimeError("Input nuclides and feature extractor must have the same nuclides!")
            feature_builder = FeatureBuilder(feature_extractor)
            truth_ = []
            features_ = []
            for nuclide in self.train_input.nuclides:
                print("building features for %s \n" % nuclide)
                id_peaks = self.getPeaks(feature_builder_input.peakanalysis, nuclide)
                fe, tu = feature_builder.build(id_peaks, nuclide)
                truth_.append(tu)
                features_.append(fe)
            truth = pandas.concat(truth_, ignore_index=True)
            features = pandas.concat(features_, ignore_index=True)
            if feature_builder_input.save:
                os.makedirs(f_file.parent, exist_ok=True)
                os.makedirs(t_file.parent, exist_ok=True)
                features.to_csv(f_file, index=False)
                truth.to_csv(t_file, index=False)
        # check the feature extractor and feature table are consistant
        if feature_extractor.getFeatureLabels() != list(features.columns):
            raise RuntimeError("Feature table labels and feature extractor labels must be identical!")
        return features, truth

    def getFeatureExtractor(self, roi_definer : RoiDefinerInput):
        """ Returns feature extractor or feature definitions
        """
        # ROI definition stage
        roi_file = self.train_input.buildpath.joinpath(roi_definer.output)
        if roi_definer.skip:
            if roi_file.exists():
                print("Skipping region definer, reading definitions from %s \n" % roi_file.name)
                feature_extractor = loadXml(str(roi_file))
            else:
                raise RuntimeError("%s region of interest definition file does not exists!" % roi_file.name)
        else:
            feature_extractor = FeatureExtractorROI()
            for nuclide in self.train_input.nuclides:
                print("defining roi for %s \n" % nuclide)
                regions = self.getRegions(roi_definer, nuclide)
                fen = FeatureExtractorNuclide()
                fen.setName(nuclide)
                fen.addRegions(regions)
                feature_extractor.addNuclide(fen)
            if roi_definer.save:
                os.makedirs(roi_file.parent, exist_ok=True)
                feature_extractor.write(roi_file)
        return feature_extractor

    def getRegions(self, roi_definer : RoiDefinerInput, nuclide):
        """ Provides regions for a specific nuclide
        """
        roi_peaks = self.getPeaks(roi_definer.peakanalysis, nuclide)
        roi_peaks = [p.sample for p in roi_peaks]
        regions = defineRegions(roi_peaks, roi_definer.peakanalysis.spa.sensor,
                                snr=roi_definer.snr, limit=roi_definer.limit, min_width = roi_definer.min_width)
        return regions

    def getPeaks(self, peakanalysis : PeakAnalysisInput, nuclide) -> PeakResultsList:
        """ Provides peak results, saves results if required.
        """
        peak_file = Path("sources", nuclide, "peaks", peakanalysis.output)
        if peakanalysis.skip:
            file = self.train_input.buildpath.joinpath(peak_file)
            if file.exists():
                print("Skipping peak extraction for %s, reading from %s" % (nuclide,peak_file.name))
                peak_results = loadXml(file)
            else:
                raise RuntimeError("Peaks for %s must exist to skip peak finding step" % nuclide)
        else:
            id_inputs = self.getInputs(peakanalysis.sample_generator, nuclide)
            peak_processor = PeakProcessor(peakanalysis.spa)
            peak_results = peak_processor.analyze(id_inputs)
            if peakanalysis.save:
                file = self.train_input.buildpath.joinpath(peak_file)
                os.makedirs(file.parent, exist_ok=True)
                peak_results.write(file, compress=True)
        return peak_results

    def getInputs(self, sg_input : SampleGeneratorInput, nuclide) -> IdentificationInputList:
        """ Provides identification inputs, saves results if required
        """
        inputs_file = Path("sources", nuclide, "samples", sg_input.output)
        if sg_input.skip:
            file = self.train_input.buildpath.joinpath(inputs_file)
            if file.exists():
                print("Skipping sampling for %s, reading from %s" % (nuclide, inputs_file.name))
                id_inputs = loadXml(file)
            else:
                raise RuntimeError("Inputs for %s must exist to input generation step" % nuclide)
        else:
            path = sg_input.template.sourcepath.joinpath(nuclide, sg_input.template.filename)
            if not path.exists():
                raise RuntimeError("%s nuclide template path does not exist" % nuclide)
            else:
                templates = loadXml(path)
            samp_gen = SampleGenerator()
            id_inputs = samp_gen.getInputs(templates, sg_input.counts, sg_input.samples, sg_input.background, sg_input.fraction)
            if sg_input.save:
                file = self.train_input.buildpath.joinpath(inputs_file)
                os.makedirs(file.parent, exist_ok=True)
                id_inputs.write(file, compress=True)
        return id_inputs
