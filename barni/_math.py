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
import _barni

__all__ = ['SolveAugmentedTridiag']


# FIXME: store the tridiagonal matrix in matrix diagonal ordered form used in scipy banded solvers
def convert2tri(matrix):
    """ Converts matrix into tridiagonal with 3 rows and same number of columns.
    """
    tridiag = np.zeros((3, matrix.shape[1]))
    tridiag[0, :-1] = matrix.diagonal(-1).copy()
    tridiag[1, :] = matrix.diagonal(0).copy()
    tridiag[2, 1:] = matrix.diagonal(1).copy()
    return tridiag


class SolveAugmentedTridiag():
    """
    Solves a tridiagonal matrix augmented with full rank matricies.
    """

    def __init__(self, A11, A12, A21, A22, B1, B2):
        """
        Solves the special problem of augmented matricies were
        A11 is tridiagonal and all others are full rank. Solutions
        are stored in C1 and C2.

        B1   A11 | A12   C1
        -- = ---   --- * --
        B2   A21 | A22   C2
        """
        if 3 != A11.shape[0]:
            raise ValueError("A11 matrix must be a tridiagonal")
        if A11.shape[1] != A21.shape[1]:
            raise ValueError(
                "A11 and A21 must have the same number of columns")
        if A12.shape[1] != A22.shape[1]:
            raise ValueError(
                "A12 and A22 must have the same number of columns")
        if B1.shape[0] != A12.shape[0]:
            raise ValueError("B1 and A12 must have the same number of rows")
        if B2.shape[0] != A21.shape[0] or B2.shape[0] != A22.shape[0]:
            raise ValueError(
                "B2, A21 and A22 must have the same number of rows")
        self.A11 = A11.astype(np.double)
        self.A12 = A12.astype(np.double)
        self.A21 = A21.astype(np.double)
        self.A22 = A22.astype(np.double)
        self.B1 = B1.astype(np.double)
        self.B2 = B2.astype(np.double)
        self.C1 = np.zeros(B1.size, dtype=np.double)
        self.C2 = np.zeros(B2.size, dtype=np.double)

    def solve(self):
        self.reduceTridiag()
        self.zeroLower()
        self.solveLower()
        self.backPropagateUpperRight()
        self.solveUpper()

    def reduceTridiag(self):
        """ Reduces the tridiag matrix to a upper diag in echelon form
        """
        _barni.reduce_tridiag(self.A11, self.A12, self.B1)

    def zeroLower(self):
        """ Zero out lower left A21 matrix
        """
        _barni.zero_lower(self.A11, self.A12, self.A21,
                          self.A22, self.B1, self.B2)
        if not np.isclose(self.A21.sum(), 0):
            raise ValueError("Lower-left matrix (A21) not eliminated")

    def solveLower(self):
        B = self.B2.flatten()
        self.C2 = scipy.optimize.nnls(self.A22, B)[0]

    def backPropagateUpperRight(self):
        self.B1 -= (self.A12 @ self.C2).reshape(-1, 1)

    def solveUpper(self):
        # FIXME: store the tridiagonal matrix in matrix diagonal ordered form used in scipy banded solvers
        ab = np.zeros((2, self.A11.shape[1]))
        ab[0, 1:] = self.A11[0, :-1]
        ab[1, :] = self.A11[1]
        self.C1 = scipy.linalg.solve_banded((0, 1), ab, self.B1)


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
