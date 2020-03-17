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
'''
Module for various math routines.

@author monterial1
'''

import scipy
import numpy as np

__all__ = ['SolveAugmentedTridiag']


class SolveAugmentedTridiag():
    """
    Solves a tridiagonal matrix augmented with full rank matricies.
    """

    def __init__(self, A11, A12, A21, A22, B1, B2):
        """
        A11 is a tridiagonal matrix, all else are full rank
        Bs are vectors
        """
        self.A11 = A11
        self.A12 = A12
        self.A21 = A21
        self.A22 = A22
        self.B1 = B1
        self.B2 = B2
        self.C1 = np.zeros(B1.size)
        self.C2 = np.zeros(B2.size)
        self.toprows = A11.shape[0]
        self.bottomrows = A21.shape[0]
        self.leftcolumns = A11.shape[1]
        self.rightcolumns = A12.shape[1]

    def solve(self):
        self.reduceTridiag()
        self.zeroLower()
        self.solveLower()
        self.backPropagateUpperRight()
        self.solveUpper()

    def reduceTridiag(self):
        """ Reduces the tridiag matrix to a upper diag in echelon form
        """
        for i in range(self.toprows - 1):
            f = (self.A11[i + 1, i] / self.A11[i, i])  # (c1 / a1)
            self.A11[i + 1, :] -= self.A11[i, :] * f
            self.A12[i + 1, :] -= self.A12[i, :] * f
            self.B1[i + 1, :] -= self.B1[i, :] * f
            f = 1 / self.A11[i, i]
            self.A11[i, :] *= f
            self.A12[i, :] *= f
            self.B1[i, :] *= f
        # special treatment for last row
        i += 1;
        f = 1 / self.A11[i, i]
        self.A11[i, :] *= f
        self.A12[i, :] *= f
        self.B1[i, :] *= f

    def zeroLower(self):
        """ Zero out lower left A21 matrix
        """
        for i in range(self.toprows):  # the number of bins
            f = self.A21[:, i][:,None]
            self.A22[:, :] -= self.A12[i, :] * f
            self.B2[:, :] -= self.B1[i, :] * f
            self.A21[:, :] -= self.A11[i, :] * f
        if not np.isclose(self.A21.sum(), 0):
            raise ValueError("Lower-left matrix (A21) not eliminated")

    def solveLower(self):
        B = self.B2.flatten()
        self.C2 = scipy.optimize.nnls(self.A22, B)[0]

    def backPropagateUpperRight(self):
        self.B1 -= (self.A12 @ self.C2).reshape(-1, 1)

    def solveUpper(self):
        A_diag = np.zeros((2, self.toprows))
        A_diag[0, 1:] = self.A11.diagonal(1)
        A_diag[1, :] = self.A11.diagonal(0)
        # solve for baseline
        self.C1 = scipy.linalg.solve_banded((0, 1), A_diag, self.B1)


def gauss_pdf(x, mu, std):
    ''' Calculates the probability on a Gaussian function.

    Args:
        x: Point at which to calculate probability
        mu: Mean of the distribution
        std: Standard deviation of the distribution

    Returns:
        Probability at that point

    '''
    g = np.exp(-0.5*((x - mu)/std) ** 2) / (std * np.sqrt(2*np.pi))
    return g