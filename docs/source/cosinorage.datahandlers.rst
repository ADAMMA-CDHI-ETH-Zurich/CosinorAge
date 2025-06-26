cosinorage.datahandlers Module
=============================

Module Contents
---------------

.. automodule:: cosinorage.datahandlers
   :no-members:


Classes
-------

.. autoclass:: cosinorage.datahandlers.DataHandler
   :members:

.. autoclass:: cosinorage.datahandlers.NHANESDataHandler
   :members:

.. autoclass:: cosinorage.datahandlers.GalaxyDataHandler
   :members:

.. autoclass:: cosinorage.datahandlers.UKBDataHandler
   :members:

.. autoclass:: cosinorage.datahandlers.GenericDataHandler
   :members:

Utility Functions
-----------------

Generic Data Functions
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.datahandlers.read_generic_xD()

Galaxy Smartwatch Data Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.datahandlers.read_galaxy_binary_data()

.. autofunction:: cosinorage.datahandlers.filter_galaxy_binary_data()

.. autofunction:: cosinorage.datahandlers.resample_galaxy_binary_data()

.. autofunction:: cosinorage.datahandlers.preprocess_galaxy_binary_data()

.. autofunction:: cosinorage.datahandlers.acceleration_data_to_dataframe()

.. autofunction:: cosinorage.datahandlers.calibrate_binary()

.. autofunction:: cosinorage.datahandlers.remove_noise_binary()

.. autofunction:: cosinorage.datahandlers.detect_wear_binary()

.. autofunction:: cosinorage.datahandlers.calc_weartime_binary()

UK Biobank Data Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.datahandlers.read_ukb_data()

.. autofunction:: cosinorage.datahandlers.filter_ukb_data()

.. autofunction:: cosinorage.datahandlers.resample_ukb_data()


NHANES Data Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.datahandlers.read_nhanes_data()

.. autofunction:: cosinorage.datahandlers.filter_nhanes_data()

.. autofunction:: cosinorage.datahandlers.resample_nhanes_data()

.. autofunction:: cosinorage.datahandlers.remove_bytes()

.. autofunction:: cosinorage.datahandlers.clean_data()

.. autofunction:: cosinorage.datahandlers.calculate_measure_time()


General Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.datahandlers.filter_incomplete_days()

.. autofunction:: cosinorage.datahandlers.filter_consecutive_days()

.. autofunction:: cosinorage.datahandlers.largest_consecutive_sequence()

.. autofunction:: cosinorage.datahandlers.calculate_enmo()

.. autofunction:: cosinorage.datahandlers.calculate_minute_level_enmo()


Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.datahandlers.plot_orig_enmo()

.. autofunction:: cosinorage.datahandlers.plot_enmo()

.. autofunction:: cosinorage.datahandlers.plot_orig_enmo_freq()
   