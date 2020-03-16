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

"""
BARNI XML reader module.
"""

from xml.dom import minidom as dom
import binascii
import gzip

__all__ = ['loadXml']

readers = {}

def registerReader(tagName, readerFunction):
    readers[tagName] = readerFunction

def is_gz_file(filepath):
    with open(filepath, 'rb') as test_f:
        return binascii.hexlify(test_f.read(2)) == b'1f8b'

class ReaderContext(object):
    def __init__(self, document):
        self.document = document

    def convert(self, element):
        name = element.nodeName
        if name in readers.keys():
            reader = readers[name]
        else:
            raise ValueError("%s does not have a BARNI XML loader" % name)
        return reader(self, element)

    def getElementPath(self, element):
        out = [element.nodeName]
        while element.parentNode:
            out.insert(0, element.nodeName)
            element = element.parentNode
        return "/".join(out)

    def raiseAttributeError(self, element, p):
        raise ValueError(
            "Bad attribute %s at %s" %
            (p, self.getElementPath(element)))

    def raiseElementError(self, element, node):
        raise ValueError(
            "Bad tag %s at %s" %
            (node.nodeName, self.getElementPath(element)))


def loadXml(filename):
    """ Function for loading in BARNI classes written to XML.

    Args:
        filename: String name of the file

    Returns:
        BARNI Object loaded from XML
    """
    if is_gz_file(filename):
        with gzip.GzipFile(str(filename), "rb") as file:
            document = dom.parse(file)
    else:
        document = dom.parse(str(filename))
    context = ReaderContext(document)
    root = document.documentElement
    return context.convert(root)
