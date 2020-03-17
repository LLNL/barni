Command Line Interface
======================
BARNI provides a command line interface for both identification [id] and training [train]. 
Sample file configuration files can be found in :file:`examples/` directory. Both routines require .yml formatted configuration files as positional argument. The following is printed withe the 
use of the [-h] flag:

usage: barni_cli.py [-h] [-i INPUT] [-o OUTPUT] [-p] {id,train} config

BARNI Command Line Interface

positional arguments:
  {id,train}            choice of either identify or training routine
  config                yaml configuration file for identification or training

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Barni identification input file
  -o OUTPUT, --output OUTPUT
                        Identification results output file
  -p, --plot            Plot the results of identification

Identification
--------------
The :py:class:`identification configuration <barni._id.IdentificationAlgorithmInput>` file contains references to serialized versions: 

  1. :py:class:`peakanalysis <barni._architecture.PeakAnalysis>`
  2. :py:class:`feature_extractor <barni._architecture.FeatureExtractor>`
  3. :py:class:`classifier <barni._architecture.Classifier>`

The first is meant to be manipulated by a user, and provides parameters for peak
search and information about the sensor. The second and the third files are generated
from the training routine. 

The identification routine requires an input [-i], which should is a serialized 
:py:class:`IdentificationInput <barni._architecture.IdentificationInput>`. 
The output [-o] is optional and by be default it is written to a :file:`id_results.xml` 
file. The output file contains a list nuclide results which are also printed 
to the screen. The spectrum plot of the results can be provided with the optional 
[-p] flag. 

Training
--------
The training routine requires the :py:class:`training configuration 
<barni._training.TrainingInput>` file.
The training routine produces two outputs required for identification the 
feature extractor (regions of interest) and classifier. 

The training routine requires :py:class:`TemplateList <barni._spectrum.TemplateList>`
for each nuclide of interest. A template file must exist for every nuclide specified
in the confgiguration file. The :file:`templates` directory, shown below,
should contain a folder for each nuclide with the templates file inside. Both the
directory and the file name are specified in the configuration file. 

.. parsed-literal::
  templates 
   \|-- Cs137
   |    \|-- :py:class:`templates.xml.gz <barni._spectrum.TemplateList>`
   \|-- Co60
        \|-- :py:class:`templates.xml.gz <barni._spectrum.TemplateList>`

The :file:`build` directory directory stores the output of each file. The tree-structure
below is linked to the BARNI classes. Notice that both the classifier (training) and feature 
extractor routines (roi) produced samples and peaks which are saved under their respective
folders for each nuclide. 

.. parsed-literal::
  :file:`build`
  \|-- sources
  |    \|-- Cs137
  |         \|-- samples
  |         \|    \|-- :py:class:`roi_samples.xml.gz <barni._id.IdentificationInputList>`
  |         \|    +-- :py:class:`training_samples.xml.gz <barni._id.IdentificationInputList>`
  |         \|-- peaks
  |              \|-- :py:class:`roi_peaks.xml.gz <barni._peak.PeakResultsList>`
  \|              +-- :py:class:`training_peaks.xml.gz <barni._peak.PeakResultsList>`
  \|            
  \|-- :py:class:`roi.xml <barni._architecture.FeatureExtractor>`
  \|-- truth.csv.gz
  \|-- features.csv.gz
  \|-- :py:class:`classifiers.pic <barni._architecture.Classifier>`

The classification routine can optionally produce the feature and truth tables. These
are stored in compressed comma seperated value filex :file:`feature.csv.gz` and :file:`truth.csv.gz` 
The results needed for identification in the example above are the :file:`roi.xml` 
and :file:`classifiers.pic` files. 
