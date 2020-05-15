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

import unittest
import tempfile
from barni import Spectrum, EnergyScale, loadXml, SpectrumList
import os
import warnings

class  SpectrumTestCase(unittest.TestCase):

    def setUp(self):
        self.counts = (1,1,1)
        self.edges = (0,1,2,3)
        self.energyScale = EnergyScale(self.edges)
        self.realtime = 1
        self.livetime = 1
        self.distance = 10.
        self.gamma_dose = 100.
        self.title = "test"
        self.spectrum = Spectrum(self.counts,
                                self.energyScale,
                                self.realtime,
                                self.livetime,
                                self.distance,
                                self.gamma_dose,
                                self.title)

    def test_Integral(self):
        self.assertEqual(self.spectrum.getIntegral(0,3), 3)
        self.assertEqual(self.spectrum.getIntegral(0.6, 3), 2.4)
        self.assertEqual(self.spectrum.getIntegral(0, 2.4), 2.4)
        # supress the out-of-bounds warning thrown from using -1 energy
        warnings.simplefilter('ignore', category=UserWarning)
        self.assertEqual(self.spectrum.getIntegral(-1, 4), 3)
        warnings.simplefilter('default', category=UserWarning)


    def test_downsample(self):
        sp2 = self.spectrum.downsample()
        self.assertSequenceEqual(tuple(sp2.energyScale.getEdges()), (0, 2))
        self.assertSequenceEqual(tuple(sp2.counts), (2, 2))

    def test_copy(self):
        sp2 = self.spectrum.copy()
        self.assertSequenceEqual(tuple(sp2.energyScale.getEdges()), self.edges)
        self.assertSequenceEqual(tuple(sp2.counts), self.counts)
        self.assertEqual(sp2.realtime, self.realtime)
        self.assertEqual(sp2.livetime, self.livetime)
        self.assertEqual(sp2.distance, self.distance)
        self.assertEqual(sp2.gamma_dose, self.gamma_dose)
        self.assertEqual(sp2.title, self.title)

    def test_getNormedCounts(self):
        self.assertSequenceEqual(tuple(self.spectrum.getNormedCounts()), (1,1,1))
        sp2 = Spectrum([1,1,1,1,1], EnergyScale([0,1,2,4,8,8.5]))
        self.assertSequenceEqual(tuple(sp2.getNormedCounts()), (1., 1., 0.5, 0.25, 2.))

    def test_toXml(self):
        """ Write and read to temporaty file and compare
        """
        tmp = os.environ.get("TEMP", "build/test")
        os.makedirs(tmp, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=tmp) as fp:
            self.spectrum.write(fp.name)
            sp2 = loadXml(fp.name)
            self.assertSequenceEqual(tuple(sp2.energyScale.getEdges()), self.edges)
            self.assertSequenceEqual(tuple(sp2.counts), self.counts)
            self.assertEqual(sp2.realtime, self.realtime)
            self.assertEqual(sp2.livetime, self.livetime)
            self.assertEqual(sp2.distance, self.distance)
            self.assertEqual(sp2.gamma_dose, self.gamma_dose)
            self.assertEqual(sp2.title, self.title)

class  SpectrumListTestCase(unittest.TestCase):

    def test_toXml(self):
        counts = (1,1,1)
        edges = (0,1,2,3)
        energyScale = EnergyScale(edges)
        rt = 1
        lt = 1
        distance = 10.
        gamma_dose = 100.
        title = "test"
        spectrum = Spectrum(counts,
                            energyScale,
                            rt,
                            lt,
                            distance,
                            gamma_dose,
                            title)
        sp_list = SpectrumList()
        sp_list.addSpectrum(spectrum)
        sp_list.addSpectrum(spectrum)
        tmp = os.environ.get("TEMP", "build/test")
        os.makedirs(tmp, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=tmp) as fp:
            sp_list.write(fp.name)
            sp_list2 = loadXml(fp.name)
            for sp in sp_list2:
                self.assertSequenceEqual(tuple(sp.energyScale.getEdges()), edges)
                self.assertSequenceEqual(tuple(sp.counts), counts)
                self.assertEqual(sp.realtime, rt)
                self.assertEqual(sp.livetime, lt)
                self.assertEqual(sp.distance, distance)
                self.assertEqual(sp.gamma_dose, gamma_dose)
                self.assertEqual(sp.title, title)


if __name__ == "main":
    unittest.main()

