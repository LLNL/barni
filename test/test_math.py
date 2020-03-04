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
import numpy as np
from barni import SolveAugmentedTridiag
from barni._math import gauss_pdf, convert2tri

class  Convert2TriTestCase(unittest.TestCase):

    def setUp(self):
        A = np.zeros((5, 5))
        for i in range(5):
            for j in range(i - 1, i + 2):
                if j < 5 and j >= 0:
                    A[i, j] = 1 + j
        self.A = convert2tri(A)

    def test_Main(self):
        self.assertSequenceEqual(tuple(self.A[1,:]), (1,2,3,4,5))

    def test_Upper(self):
        self.assertSequenceEqual(tuple(self.A[0,:]), (1,2,3,4,0))

    def test_Lower(self):
        self.assertSequenceEqual(tuple(self.A[2,:]), (0,2,3,4,5))

class  SolveAugmentedTridiagTestCase(unittest.TestCase):

    def setUp(self):
        channels = 100
        x = np.arange(channels)
        shapes = [gauss_pdf(x, 50, 1)]
        spectrum = np.ones(channels) * 10
        n1 = len(spectrum)
        S = np.array(shapes).T
        A11 = np.zeros((3, n1))
        c2 = 0
        # FIXME: store the tridiagonal matrix in matrix diagonal ordered form used in scipy banded solvers
        for i in range(0, n1 - 1):
            c = i * 1.
            A11[1, i] = 1 + c + c2
            A11[0, i] = -c  # upper
            A11[2, i + 1] = -c  # lower
            c2 = c
        A11[1, -1] = 1 - c2
        # Populate the unfolding portion
        B1 = np.array([spectrum]).T
        B2 = S.T @ B1
        A21 = S.T.copy()
        A22 = S.T @ S
        A12 = S
        self.solver = SolveAugmentedTridiag(A11, A12, A21, A22, B1, B2)
        self.solver.solve()

    def test_reduceTriag(self):
        self.assertAlmostEqual(self.solver.A11[2,:].sum(), 0.)

    def test_zeroLower(self):
        self.assertAlmostEqual(self.solver.A21.sum(), 0.)

    def test_solveLower(self):
        self.assertAlmostEqual(self.solver.C2[0], 0.1781282200451073)

    def test_PropagateUpperRight(self):
        self.assertAlmostEqual(self.solver.B1.sum(), 161.29553683318667)

    def test_solveUppwer(self):
        self.assertAlmostEqual(self.solver.C1.sum(), 892.4080570507986)

if __name__ == "main":
    unittest.main()
