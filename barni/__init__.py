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

from . import _architecture
from . import _id
from . import _peak
from . import _bins
from . import _class
from . import _reader
from . import _fe
from . import _roi
from . import _spectrum
from . import _training
from . import _unfolder
from . import _plot
from . import _smoothing
from . import _spa
from . import _sensor
from . import _label
from . import _result
from . import _math
from . import physics


# Import the public symbols
from ._architecture import *
from ._id import *
from ._peak import *
from ._class import *
from ._unfolder import *
from ._fe import *
from ._reader import *
from ._spectrum import *
from ._bins import *
from ._plot import *
from ._roi import *
from ._training import *
from ._smoothing import *
from ._sensor import *
from ._spa import *
from ._label import *
from ._result import *
from ._math import *

# Export the public symbols
__all__ = []
__all__.extend(_architecture.__all__)
__all__.extend(_bins.__all__)
__all__.extend(_id.__all__)
__all__.extend(_peak.__all__)
__all__.extend(_reader.__all__)
__all__.extend(_fe.__all__)
__all__.extend(_roi.__all__)
__all__.extend(_smoothing.__all__)
__all__.extend(_spa.__all__)
__all__.extend(_spectrum.__all__)
__all__.extend(_sensor.__all__)
__all__.extend(_unfolder.__all__)
__all__.extend(_class.__all__)
__all__.extend(_training.__all__)
__all__.extend(_label.__all__)
__all__.extend(_plot.__all__)
__all__.extend(_result.__all__)
__all__.extend(_math.__all__)

__version_info__ = (0,1,1)
__version__ = ".".join(str(i) for i in __version_info__)