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
Command line interface for BARNI

@author: monterial1
'''

import argparse
import sys

from barni._id import IdentificationAlgorithm
from barni._training import TrainProcessor, TrainingInput
from barni._id import IdentificationAlgorithmInput
from barni._plot import plotPeakResult
from barni._reader import loadXml

class MyParser(argparse.ArgumentParser):
    """ Custom parser
    """
    def error(self, message):
        """ Print error message
        """
        self.print_help()
        sys.stderr.write('\n ERROR: %s\n' % message)
        sys.exit(2)

    def setArgs(self):
        """ Setup all the arguments
        """
        self.add_argument("run", choices=["id", "train"], type=str,
                            help="choice of either identify or training routine")
        self.add_argument("config", type=str, help="yaml configuration file for identification or training")
        self.add_argument("-i", "--input", type=str, help="Barni identification input file")
        self.add_argument("-o", "--output", type=str, help="Identification results output file",
                            default="id_results.xml")
        self.add_argument("-p", "--plot", action='store_true', help="Plot the results of identification")


def printBanner():
    print(" ____          _____  _   _ _____ ")
    print("|  _ \   /\   |  __ \| \ | |_   _|")
    print("| |_) | /  \  | |__) |  \| | | |  ")
    print("|  _ < / /\ \ |  _  /| . ` | | |  ")
    print("| |_) / ____ \| | \ \| |\  |_| |_ ")
    print("|____/_/    \_\_|  \_\_| \_|_____|")
    print("                                  ")


def runId(config, input, plot=False):
    """
    Args:
        config (str): Identification configuration (yml) file.
        input: Path to identification input file
        plot: Optional plotting flag
    """
    input = loadXml(input)
    id_input = IdentificationAlgorithmInput()
    id_input.fromfile(config)
    algorithm = IdentificationAlgorithm()
    algorithm.setPeakExtractor(id_input.peakanalysis)
    algorithm.setFeatureExtractor(id_input.feature_extractor)
    algorithm.setClassifier(id_input.classifiers)
    result = algorithm.identify(input)
    predictions = result.classifications.getPredictions()
    print(predictions.toXml())
    predictions.write(args.output)
    if plot:
        plotPeakResult(input, result.peaks.sample)

def runTrain(config):
    """
    Args:
        config (str): Training configuration (yml) file
    """
    train_input = TrainingInput()
    train_input.fromfile(config)
    train_procs = TrainProcessor(train_input)
    train_procs.train()

if __name__ == "__main__":
    printBanner()
    parser = MyParser(description='BARNI Command Line Interface')
    parser.setArgs()
    args = parser.parse_args()
    runtype = args.run
    if runtype.lower() == "id":
        if args.input is None:
            raise parser.error("INPUT [-i] is requred for the id routine.")
        runId(args.config, args.input, args.plot)
    elif runtype.lower() == "train":
        runTrain(args.config)
