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
from barni import EnergyScale
from barni import loadXml
import tempfile
import os

class EnergyScaleTestCase(unittest.TestCase):
    def setUp(self):
        self.energy_scale = EnergyScale([0, 1, 2, 3, 4, 5])

    def test_newScale(self):
        es = EnergyScale.newScale(0, 5, 1, 1)
        self.assertSequenceEqual(tuple(es.getEdges()), tuple(self.energy_scale.getEdges()))

    def test_getCenter(self):
        self.assertEqual(self.energy_scale.getCenter(4), 4.5)

    def test_getCenters(self):
        self.assertSequenceEqual(tuple(self.energy_scale.getCenters()), tuple([0.5,1.5,2.5,3.5,4.5]))

    def test_findBin(self):
        self.assertEqual(self.energy_scale.findBin(5), 4)
        self.assertEqual(self.energy_scale.findBin(6), 5)
        self.assertEqual(self.energy_scale.findBin(-1), 0)

    def test_Length(self):
        self.assertEqual(len(self.energy_scale), 6)

    def test_findEnergy(self):
        self.assertEqual(self.energy_scale.findEnergy(1.5), 1.5)

    def test_toXml(self):
        """ Write and read to temporaty file and compare
        """
        tmp = os.environ.get("TEMP", "build/test")
        os.makedirs(tmp, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=tmp) as fp:
            self.energy_scale.write(fp.name)
            es = loadXml(fp.name)
            self.assertSequenceEqual(tuple(es.getEdges()), tuple(self.energy_scale.getEdges()))

if __name__ == "main":
    unittest.main()

