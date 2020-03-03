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
Various math routines 

@author monterial1
'''
cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

def fill_smooth(double[:,:] A, double f):
    """ Set up the smooth tridiagonal matrix, does not assume
    Returns:
    """
    assert A.shape[0] == 3
    cdef Py_ssize_t n = A.shape[1]
    cdef Py_ssize_t i
    cdef double c2 = 0
    cdef double c = 0
    for i in range(0, n - 1):
        # Lambda will change with bin so that the filtering smoothness
        # matches the peak width
        c = i * f
        # The diagonal controls matching the inputs and
        # the off diagonals hold the smoothing
        A[1, i] = 1 + c + c2  # Diagonal
        A[0, i + 1] = -c  # Upper
        A[2, i] = -c  # Lower
        c2 = c
    i += 1
    A[1, i] = 1 + c2
    A[2, i] = 0

def zero_lower(double[:,:] A11, double[:,:] A12, double[:,:] A21, double[:,:] A22, double[:,:] B1, double[:,:] B2):
    """ Zero out lower left A21 matrix, assuming A11 has been reduced to row echelon form.

    The problem is formed from augmented matricies:
        B1   A11 | A12   C1
        -- = ---   --- * --
        B2   A21 | A22   C2
    """
    assert A21.shape[1] == A12.shape[0]
    cdef Py_ssize_t  leftcols = A21.shape[1]
    cdef Py_ssize_t bottomrows = A21.shape[0]
    cdef Py_ssize_t rightcols = A22.shape[1]
    cdef Py_ssize_t i, j
    cdef double f
    for i in range(leftcols):  # the number of total column
        for j in range(bottomrows):  # bottom rows
            f = A21[j, i]
            for k in range(rightcols):  # all right columns
                A22[j, k] -= A12[i, k] * f
            B2[j, 0] -= B1[i, 0] * f
            if leftcols - 1 > i:
                A21[j, i + 1] -= A11[0, i] * f
            A21[j, i] -= A11[1, i] * f

def reduce_tridiag(double[:,:] A11, double[:,:]A12, double[:,:] B1):
    """ Reduces tridiag matrix into echelon form
    Args:
        A11: tridiag matrix, with 3 rows!
        A12: matrix augmented with tridiag
        B1: left hand side

    Returns:
        Transformed version of all three.
    """

    cdef Py_ssize_t rows = A11.shape[0]
    assert rows == 3
    cdef Py_ssize_t cols = A11.shape[1]
    cdef Py_ssize_t rows2 = A12.shape[0]
    assert cols == rows2
    cdef Py_ssize_t cols2 = A12.shape[1]
    cdef Py_ssize_t i, j
    cdef double f
    for i in range(cols - 1):
        # remove bottom diagonal
        f = (A11[2, i+1] / A11[1,i]) # c1 / a1`
        for j in range(cols2):
            A12[i + 1, j] -= A12[i, j] * f
        B1[i + 1, 0] -= B1[i, 0] * f
        A11[1, i + 1] -= A11[0, i] * f  # a2 - b1 * f
        # mutate c1 last
        A11[2, i+1] -= A11[1, i] * f # c1 - a1 * f
        # normalize diagonal
        f = 1. / A11[1, i] # 1 / a1
        for j in range(cols2):
            A12[i, j] *= f
        B1[i, 0] *= f
        A11[0,i] *= f # change b1
        A11[1,i] *= f # change a1
    # special treatment for last row
    i += 1;
    f = 1. / A11[1, i]
    for j in range(cols2):
        A12[i, j] *= f
    B1[i, 0] *= f
    A11[1,i] *= f
