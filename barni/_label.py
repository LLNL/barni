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

"""
Label module.

Labels are usually attached to templates which have known nuclides and shielding configurations.
"""


__all__ = ["Label"]

from ._reader import registerReader
from ._architecture import Serializable

class Label(Serializable):
    """
    This class is attached to results to create labeled data.

    The label will contain
      nuclide - the name of the nuclide this belongs to
      z - the atomic number of the shielding
      ad - the thickness of the shielding in g/cm^2

    """

    def __init__(self):
        self.name = ""
        self.z = 0
        self.ad = 0

    def __str__(self):
        return "%s (%d,%5.2f)" % (self.name, self.z, self.ad)

    def toXml(self):
        """ Converts label to XML string
        """

        xml = "<Label>\n"
        attributes = ["name", "ad", "z"]
        for attr in attributes:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if value is not None:
                    xml += "  <%s>" % attr
                    xml += str(value)
                    xml += "</%s>\n" % attr
        xml += "</Label>\n"
        return xml

def loadLabel(context, element):
    """
    Converts a dom element into a FeatureExtractor object
    """

    floatFields = ['ad','z']
    stringFields = ['name']

    out = Label()
    for node in element.childNodes:
        # skip all but elements
        if node.nodeType != node.ELEMENT_NODE:
            continue
        if node.tagName in floatFields:
            out.__setattr__(node.tagName, float(node.firstChild.nodeValue))
            continue
        if node.tagName in stringFields:
            out.__setattr__(node.tagName, str(node.firstChild.nodeValue))
            continue
        context.raiseElementError(element, node)
    return out

registerReader("Label", loadLabel)