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
import os
import numpy as np
import pathlib

import barni._spa as spa
from barni import Peak, RegionOfInterest, EnergyScale, Spectrum, GaussianSensorModel, IdentificationInput, loadXml

class SmoothPeakResultTestCase(unittest.TestCase):
    def test_roi(self):
        peak = Peak(600, 1, 0, 10)
        peakresult = spa.SmoothPeakResult([peak], None, None)
        roi = RegionOfInterest(590, 610)
        peak = peakresult.getRegionOfInterest(roi)
        self.assertAlmostEqual(peak.energy, 600)
        self.assertAlmostEqual(peak.intensity, 0.68268949)
        roi = RegionOfInterest(600, 99999999999999)
        peak = peakresult.getRegionOfInterest(roi)
        self.assertAlmostEqual(peak.intensity, 0.5)


class SmoothPeakAnalysisTestCase(unittest.TestCase):
    def setUp(self):
        self.sensor = GaussianSensorModel(0.05)
        energies = [40, 116, 130] # 3 total peaks
        self.peaks = []
        for erg in energies:
            self.peaks.append(Peak(erg, 1000, 0))
        channels = np.arange(0, 100)
        es = EnergyScale(np.arange(0, 303, 3))  # 3 keV per channel
        counts = channels * -0.5 + 100
        for peak in self.peaks:
            response = self.sensor.getResponse(peak.energy, peak.intensity, es.getEdges())
            counts += response
        self.spectrum = Spectrum(counts, es)
        self.mu = 1
        self.lld = 40
        self.baseline, self.smooth = spa.computeBaseline(self.spectrum.counts, self.mu)
        self.responsed_peaks = spa.responsePeaks(self.peaks, self.sensor, self.spectrum.energyScale)
        self.id_input = IdentificationInput(self.spectrum, intrinsic=self.spectrum)
        self.analysis = spa.SmoothPeakAnalysis()
        self.analysis.setSensor(self.sensor)
        self.result = self.analysis.analyze(self.id_input)

    def test_computeBaseline(self):
        self.assertAlmostEqual(self.baseline.sum(), 9076.84633432)
        self.assertAlmostEqual(self.spectrum.counts.sum(), self.smooth.sum())

    def test_getInitialPeaks(self):
        peaks = spa.getInitialPeaks(self.spectrum.counts, self.baseline, self.spectrum.energyScale, self.sensor, self.lld, self.mu)
        self.assertEqual(1, len(peaks))
        peak = peaks[0]
        self.assertAlmostEqual(peak.channel, 40.4523350)
        self.assertAlmostEqual(peak.energy, 121.3570051)
        self.assertAlmostEqual(peak.intensity, 106.8031088)

    def test_responsePeaks(self):
        self.assertEqual(len(self.responsed_peaks), 3)
        for p in self.responsed_peaks:
            self.assertEqual(p.response.sum(), 1.0)

    def test_addNeighborPeaks(self):
        energy_answers = [37.3197810, 40, 42.6802190, 111.0161278, 116, 120.9838722, 124.6691709, 130, 135.3308291]
        peaks = spa.addNeighborPeaks(self.peaks, self.sensor)
        for p, erg in zip(peaks, energy_answers):
            self.assertAlmostEqual(p.energy, erg)

    def test_combineNeighborPeaks(self):
        peaks = spa.combineNeighborPeaks(self.responsed_peaks, self.spectrum.energyScale)
        self.assertEqual(len(peaks), 1) # 3 peaks should combine to 1
        peak = peaks[0]
        self.assertAlmostEqual(peak.energy, 95.3333333)
        self.assertAlmostEqual(peak.intensity, 3000)
        self.assertAlmostEqual(peak.width, 8.5959706)

    def test_solve(self):
        baseline, intensity, peaks, response = spa.solve(self.spectrum, self.responsed_peaks, self.sensor,
                                                         self.spectrum.energyScale, self.mu, self.lld)
        intensity_answers = [0.0000000, 1043.3593481, 1028.8387752]
        energy_answers = [40, 116, 130]
        width_answers = [2.6802190, 4.9838722, 5.3308291]
        self.assertAlmostEqual(baseline.sum(), 7863.6178504)
        self.assertAlmostEqual(response.sum(), 3)
        self.assertEqual(len(peaks), 3)
        self.assertEqual(len(intensity), 3)
        for i in range(3):
            self.assertAlmostEqual(peaks[i].energy, energy_answers[i])
            self.assertAlmostEqual(peaks[i].width, width_answers[i])
            self.assertAlmostEqual(peaks[i].intensity, intensity_answers[i])
            self.assertAlmostEqual(peaks[i].intensity, intensity_answers[i])

    def test_analyze(self):
        for result in [self.result.sample, self.result.intrinsic]:
            self.assertEqual(result.getEnergyScale(), self.spectrum.energyScale)
            peaks = result.getPeaks()
            self.assertEqual(len(peaks), 2)
            self.assertAlmostEqual(peaks[0].energy, 40.0564342218949)
            self.assertAlmostEqual(peaks[0].intensity, 1079.747359800902)
            self.assertAlmostEqual(peaks[0].baseline, 178.3975619481992)
            self.assertAlmostEqual(peaks[0].width, 3.166169828312557)
            self.assertAlmostEqual(peaks[1].energy, 120.6049336569013)
            self.assertAlmostEqual(peaks[1].intensity, 1512.5442104365186)
            self.assertAlmostEqual(peaks[1].baseline, 551.5212795422568)
            self.assertAlmostEqual(peaks[1].width, 8.244610032612684)
            continuum_sum = result.getContinuum().counts.sum()
            self.assertAlmostEqual(continuum_sum, 7339.874716995713)

    def test_analyze_downsample(self):
        out = self.analysis.analyze(self.id_input, downsample=True)
        for result in [out.sample, out.intrinsic]:
            peaks = result.getPeaks()
            continuum_sum = result.getContinuum().counts.sum()
            self.assertAlmostEqual(peaks[0].energy, 38.2250329)
            self.assertAlmostEqual(peaks[0].intensity, 904.2119303)
            self.assertAlmostEqual(peaks[0].baseline, 235.0868462)
            self.assertAlmostEqual(peaks[0].width, 3.2847080)
            self.assertAlmostEqual(peaks[1].energy, 120.7943184)
            self.assertAlmostEqual(peaks[1].intensity, 1527.9884988)
            self.assertAlmostEqual(peaks[1].baseline, 545.9802068)
            self.assertAlmostEqual(peaks[1].width, 8.2010197)
            self.assertAlmostEqual(continuum_sum, 7430.0220049)

    def test_result_toXml(self):
        os.makedirs("build/test", exist_ok=True)
        with open("build/test/spectrum.test", "w") as fp:
            self.result.write(fp.name)
            r = loadXml(fp.name)
            attributes = ["sample", "intrinsic"]
            for attr in attributes:
                original = getattr(self.result, attr)
                loaded = getattr(r, attr)
                self.assertEqual(original.getEnergyScale(), loaded.getEnergyScale())
                self.assertAlmostEqual(original.getContinuum().counts.sum(), loaded.getContinuum().counts.sum())
                peaks_original = original.getPeaks()
                peaks_loaded = original.getPeaks()
                num_peaks = len(peaks_original)
                self.assertEqual(len(peaks_loaded), num_peaks)
                for i in range(len(peaks_loaded)):
                    self.assertAlmostEqual(peaks_original[i].energy, peaks_loaded[i].energy)
                    self.assertAlmostEqual(peaks_original[i].intensity, peaks_loaded[i].intensity)
                    self.assertAlmostEqual(peaks_original[i].baseline, peaks_loaded[i].baseline)
                    self.assertAlmostEqual(peaks_original[i].width, peaks_loaded[i].width)

    def test_getFit(self):
        fit = self.result.sample.getFit()
        self.assertAlmostEqual(fit.counts.sum(), 9932.1662872)
        self.assertEqual(fit.energyScale, self.result.sample.getEnergyScale())

    def test_load(self):
        data_path = pathlib.Path(__file__).parent.joinpath("data","test_spa","spa.xml")
        spa = loadXml(data_path)
        self.assertEqual(spa.startEnergy,  45.0)
        self.assertEqual(spa.endEnergy, 3000.0)
        self.assertEqual(spa.smoothingFactor, 3.0)
        sensor = spa.sensor
        self.assertEqual(sensor.resolution, 0.055)
        self.assertEqual(sensor.resolutionEnergy, 662)
        self.assertEqual(sensor.wideningPower, 0.5)
        self.assertEqual(sensor.electronicNoise, 0.0)
        self.assertEqual(sensor.C, 0.5)
        self.assertAlmostEqual(sensor.B, 0.3610784300467453)

if __name__ == "main":
    unittest.main()

