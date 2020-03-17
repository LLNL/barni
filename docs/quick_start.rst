Quick Start
===========
This quick start guide shows how to use BARNI python package for identification
and training. Sample files can be found in :file:`examples` directory.

Identification
--------------
Identification requires three files 
(1) :py:class:`peak analysis <barni._architecture.PeakAnalysis>` configuration file 
(2) :py:class:`feature extractor <barni._architecture.FeatureExtractor>`  and 
(3) :py:class:`classifier <barni._architecture.Classifier>`. The first is a user configurable
file that is easy to manipulate and requires only basic information about the sensor,
such as the energy resolution. The other two files are created from the training
routine. 

.. code:: python

    import barni
    import pickle
    # load the inputs
    spa = barni.loadXml("spa.xml")
    featureExtractor = barni.loadXml("roi_flir.xml")
    cls = barni.Classifier.load("classifiers_flir_bkg.pic")
    # load the algorithm
    algorithm = barni.IdentificationAlgorithm()
    algorithm.setPeakExtractor(spa)
    algorithm.setFeatureExtractor(featureExtractor)
    algorithm.setClassifier(cls)
    # perform identification
    input = barni.loadXml("id_input.xml")
    result = algorithm.identify(input)
    # print the results
    predictions = result.classifications.getPredictions()
    print(predictions.toXml())

Training
--------
The objective of the training is to produce the feature extractor and the classifier. 
