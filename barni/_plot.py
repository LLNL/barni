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

""" This module holds custom scaled and related functions use in
plotting spectra for barni.
"""


import numpy as np
from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models.widgets import Tabs, Panel
from bokeh.models import HoverTool

__all__ = ["plotPeakResult"]

def plotPeakResult(id_input, peakresult, outputfile="result.html"):
    """
    Plots the peak results.

    Args:
        id_input (IdentificationInput): Identification input.
        peakresult (PeakResults): Peak result
        outputfile (str): The html file coontaining the plot
    """

    energyScale = peakresult.getFit().energyScale
    centers = energyScale.getCenters()
    fit = peakresult.getFit().counts
    continuum = peakresult.getContinuum().counts
    hover = HoverTool()
    hover.mode = 'mouse'  # activate hover by vertical line
    hover.tooltips = [("energy", "@energy"),
                      ("intensity", "@intensity"),
                      ("baseline", "@baseline"),
                      ("width", "@width")]
    hover.renderers = []
    panels = []
    for axis_type in ["linear", "log"]:
        fig = figure(title="Peak Fitting Results",
                     y_axis_type=axis_type,
                     x_axis_label='Energy (keV)',
                     y_axis_label='Counts')
        fig.varea(x=centers, y1=continuum, y2=fit, color="green", legend_label="peaks")

        for peak in peakresult.getPeaks():
            ymax = fit[energyScale.findBin(peak.energy)]
            source = ColumnDataSource(data=dict(x=[peak.energy, peak.energy], y=[10e-10, ymax],
                                                energy=["%6.1f" % peak.energy] * 2,
                                                intensity=["%d" % peak.intensity] * 2,
                                                baseline=["%d" % peak.baseline] * 2,
                                                width=["%6.2f" % peak.width] * 2))
            pline = fig.line('x', 'y', color="red", source=source)
            hover.renderers.append(pline)
        fig.add_tools(hover)
        fig.varea(x=centers, y1=np.ones(continuum.size) * 10e-10, y2=continuum, color="blue", legend_label="continuum")
        fig.line(centers, id_input.sample.counts, legend_label="sample",
                 line_dash='dashed', color="black", line_width=2)
        panel = Panel(child=fig, title=axis_type)
        panels.append(panel)
    tabs = Tabs(tabs=panels)
    # add a line renderer with legend and line thickness
    # show the results
    output_file(outputfile)
    show(tabs)
