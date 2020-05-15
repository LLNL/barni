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
from barni import RegionOfInterest, Peak
from barni._architecture import NuclideResult
from barni import loadXml
import os
import tempfile



class  RegionOfInterestTestCase(unittest.TestCase):
    def setUp(self):
        self.roi = RegionOfInterest(0, 1)

    def test_Contains(self):
        self.assertTrue(0.5 in self.roi)
        self.assertFalse(2 in self.roi)

class  PeakTestCase(unittest.TestCase):
    def setUp(self):
        self.peak = Peak(662, 1, 0, 10)

    def test_toXml(self):
        tmp = os.environ.get("TEMP", "build/test")
        os.makedirs(tmp, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=tmp) as fp:
            self.peak.write(fp.name)
            # FIXME Loader for Peak does not exist.

    def test_Energy(self):
        self.assertEqual(self.peak.energy, 662)

class  NuclideResultTestCase(unittest.TestCase):
    def setUp(self):
        self.nr = NuclideResult("Co60", 0.9, 1)

    def test_toXml(self):
        """ Write and read to temporaty file and compare
        """
        tmp = os.environ.get("TEMP", "build/test")
        os.makedirs(tmp, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=tmp) as fp:
            self.nr.write(fp.name)
            nr = loadXml(fp.name)
            self.assertEqual(nr.score, 0.9)
            self.assertEqual(nr.prediction, 1)

if __name__ == "main":
    unittest.main()
