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

import unittest
import barni._spa as spa
from barni import Peak, RegionOfInterest, EnergyScale, Spectrum, GaussianSensorModel
import numpy as np

class SmoothPeakResultTestCase(unittest.TestCase):
    def setUp(self):
        # create a synthetic result
        peak = Peak(600, 1, 0, 10)
        self.peakresult = spa.SmoothPeakResult([peak], None, None)
        # create synthetic spectra
        # setup a generic spectrum with one peak
        self.sensor = GaussianSensorModel(10)
        channels = np.arange(0,100)
        linear = channels * -0.1 + 100
        std = 2
        loc = 50
        peak = np.exp(-0.5 * ((channels - loc) / std) ** 2) / (std * np.sqrt(2 * np.pi)) * 100
        counts = linear + peak
        es = EnergyScale(np.arange(0, 303,3)) # 3 keV per channel
        self.spectrum = Spectrum(counts, es)

    def test_roi(self):
        roi = RegionOfInterest(590, 610)
        peak = self.peakresult.getRegionOfInterest(roi)
        self.assertAlmostEqual(peak.energy, 600, places=7)
        self.assertAlmostEqual(peak.intensity, 0.68268949, places=7)
        roi = RegionOfInterest(600, 99999999999999)
        peak = self.peakresult.getRegionOfInterest(roi)
        self.assertAlmostEqual(peak.intensity, 0.5, places=7)

    def test_computeBaseline(self):
        baseline, smooth = spa.computeBaseline(self.spectrum.counts, mu=1)
        self.assertAlmostEqual(baseline.sum(), 9547.53704528, places=7)
        self.assertAlmostEqual(self.spectrum.counts.sum(), smooth.sum())

    def test_getInitialPeaks(self):
        baseline, smooth = spa.computeBaseline(self.spectrum.counts, mu=1)
        peaks = spa.getInitialPeaks(self.spectrum.counts, baseline, self.spectrum.energyScale, sensor=self.sensor, lld=10, mu=1)
        self.assertEqual(1, len(peaks))
        peak = peaks[0]
        self.assertAlmostEqual(peak.channel, 50.0008601, places=7)
        self.assertAlmostEqual(peak.energy, 150.0025804, places=7)
        self.assertAlmostEqual(peak.intensity, 10.0720293, places=7)


if __name__ == "main":
    unittest.main()

