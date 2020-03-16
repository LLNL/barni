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
from barni._spa import SmoothPeakResult
from barni import Peak, RegionOfInterest

class SmoothPeakResultTestCase(unittest.TestCase):
    def setUp(self):
        peak = Peak(600, 1, 0, 10)
        self.spa = SmoothPeakResult([peak], None, None)

    def test_roi(self):
        roi = RegionOfInterest(590, 610)
        peak = self.spa.getRegionOfInterest(roi)
        self.assertAlmostEqual(peak.energy, 600, places=7)
        self.assertAlmostEqual(peak.intensity, 0.682689492137, places=7)
        roi = RegionOfInterest(600, 99999999999999)
        peak = self.spa.getRegionOfInterest(roi)
        self.assertAlmostEqual(peak.intensity, 0.5, places=7)

if __name__ == "main":
    unittest.main()

