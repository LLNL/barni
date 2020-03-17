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
Module for handling the results of identification.

@author monterial1
'''

from typing import List
from collections import UserList
import textwrap

from ._architecture import NuclideResult, Serializable
from ._reader import registerReader

__all__ = ["NuclideResultList"]

class NuclideResultList(Serializable, UserList):
    """ List of nuclide results.
    """

    def addNuclideResult(self, input : NuclideResult):
        self.data.append(input)

    def toXml(self):
        xml = "<NuclideResultList>\n"
        for tmp in self.data:
            xml += textwrap.indent(tmp.toXml(), "  ")
        xml += "</NuclideResultList>\n"
        return xml

def loadNuclideResult(context, element):
    """ Loads in a nuclide result
    """
    out = NuclideResult(nuclide = None, score = None, prediction = None)
    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName == "nuclide":
            out.nuclide = str(node.firstChild.nodeValue)
            continue
        if node.tagName == "score":
            out.score = float(node.firstChild.nodeValue)
            continue
        if node.tagName == "prediction":
            out.prediction = int(node.firstChild.nodeValue)
            continue
        context.raiseElementError(element, node)
    return out

def loadNuclideResultList(context, element):
    """ Loads a list of nuclide results
    """
    out = NuclideResultList()
    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName == "NuclideResult":
            out.addTemplate(loadNuclideResult(context, node))
            continue
        context.raiseElementError(element, node)
    return out

registerReader("NuclideResult", loadNuclideResult)
registerReader("NuclideResultList", loadNuclideResultList)
