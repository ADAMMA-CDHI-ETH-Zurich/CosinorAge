cosinorage.datahandlers Module
=============================

Module Contents
---------------

.. automodule:: cosinorage.datahandlers
   :no-members:


Classes
-------

.. autoclass:: cosinorage.datahandlers.datahandler.DataHandler
   :members:

.. autoclass:: cosinorage.datahandlers.nhanesdatahandler.NHANESDataHandler
   :members:

.. autoclass:: cosinorage.datahandlers.galaxydatahandler.GalaxyDataHandler
   :members:

.. autoclass:: cosinorage.datahandlers.ukbdatahandler.UKBDataHandler
   :members:

.. autoclass:: cosinorage.datahandlers.genericdatahandler.GenericDataHandler
   :members:

Utility Functions
-----------------

Generic Data Functions
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.datahandlers.utils.generic.read_generic_xD_data

.. autofunction:: cosinorage.datahandlers.utils.generic.filter_generic_data

.. autofunction:: cosinorage.datahandlers.utils.generic.resample_generic_data

.. autofunction:: cosinorage.datahandlers.utils.generic.preprocess_generic_data

Galaxy Smartwatch Data Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.datahandlers.utils.galaxy_binary.read_galaxy_binary_data

.. autofunction:: cosinorage.datahandlers.utils.galaxy_binary.filter_galaxy_binary_data

.. autofunction:: cosinorage.datahandlers.utils.galaxy_binary.resample_galaxy_binary_data

.. autofunction:: cosinorage.datahandlers.utils.galaxy_binary.preprocess_galaxy_binary_data

.. autofunction:: cosinorage.datahandlers.utils.galaxy_binary.acceleration_data_to_dataframe

.. autofunction:: cosinorage.datahandlers.utils.galaxy_csv.read_galaxy_csv_data

.. autofunction:: cosinorage.datahandlers.utils.galaxy_csv.filter_galaxy_csv_data

.. autofunction:: cosinorage.datahandlers.utils.galaxy_csv.resample_galaxy_csv_data

.. autofunction:: cosinorage.datahandlers.utils.galaxy_csv.preprocess_galaxy_csv_data

UK Biobank Data Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.datahandlers.utils.ukb.read_ukb_data

.. autofunction:: cosinorage.datahandlers.utils.ukb.filter_ukb_data

.. autofunction:: cosinorage.datahandlers.utils.ukb.resample_ukb_data

NHANES Data Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.datahandlers.utils.nhanes.read_nhanes_data

.. autofunction:: cosinorage.datahandlers.utils.nhanes.filter_and_preprocess_nhanes_data

.. autofunction:: cosinorage.datahandlers.utils.nhanes.resample_nhanes_data

.. autofunction:: cosinorage.datahandlers.utils.nhanes.remove_bytes

.. autofunction:: cosinorage.datahandlers.utils.nhanes.clean_data

.. autofunction:: cosinorage.datahandlers.utils.nhanes.calculate_measure_time

General Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.datahandlers.utils.filtering.filter_incomplete_days

.. autofunction:: cosinorage.datahandlers.utils.filtering.filter_consecutive_days

.. autofunction:: cosinorage.datahandlers.utils.filtering.largest_consecutive_sequence

.. autofunction:: cosinorage.datahandlers.utils.calc_enmo.calculate_enmo

.. autofunction:: cosinorage.datahandlers.utils.calc_enmo.calculate_minute_level_enmo

.. autofunction:: cosinorage.datahandlers.utils.calibration.calibrate_accelerometer

.. autofunction:: cosinorage.datahandlers.utils.frequency_detection.detect_frequency_from_timestamps

.. autofunction:: cosinorage.datahandlers.utils.noise_removal.remove_noise

.. autofunction:: cosinorage.datahandlers.utils.wear_detection.detect_wear_periods

.. autofunction:: cosinorage.datahandlers.utils.wear_detection.calc_weartime

Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.datahandlers.utils.visualization.plot_orig_enmo

.. autofunction:: cosinorage.datahandlers.utils.visualization.plot_enmo

.. autofunction:: cosinorage.datahandlers.utils.visualization.plot_orig_enmo_freq
   