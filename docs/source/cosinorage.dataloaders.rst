cosinorage.DataHandlers module
=============================

module contents
---------------

.. automodule:: cosinorage.DataHandlers
   :no-members:


classes
----------------------------------------

.. autoclass:: cosinorage.DataHandlers.DataHandler
   :members:

.. autoclass:: cosinorage.DataHandlers.NHANESDataHandler
   :members:

.. autoclass:: cosinorage.DataHandlers.GalaxyDataHandler
   :members:

.. autoclass:: cosinorage.DataHandlers.UKBDataHandler
   :members:

utility functions
----------------------------------------

galaxy smartwatch data functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.DataHandlers.read_galaxy_data()

.. autofunction:: cosinorage.DataHandlers.filter_galaxy_data()

.. autofunction:: cosinorage.DataHandlers.resample_galaxy_data()

.. autofunction:: cosinorage.DataHandlers.preprocess_galaxy_data()

.. autofunction:: cosinorage.DataHandlers.acceleration_data_to_dataframe()

.. autofunction:: cosinorage.DataHandlers.calibrate()

.. autofunction:: cosinorage.DataHandlers.remove_noise()

.. autofunction:: cosinorage.DataHandlers.detect_wear()

.. autofunction:: cosinorage.DataHandlers.calc_weartime()

uk biobank data functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.DataHandlers.read_ukb_data()

.. autofunction:: cosinorage.DataHandlers.filter_ukb_data()

.. autofunction:: cosinorage.DataHandlers.resample_ukb_data()


nhanes data functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.DataHandlers.read_nhanes_data()

.. autofunction:: cosinorage.DataHandlers.filter_nhanes_data()

.. autofunction:: cosinorage.DataHandlers.resample_nhanes_data()

.. autofunction:: cosinorage.DataHandlers.remove_bytes()

.. autofunction:: cosinorage.DataHandlers.clean_data()

.. autofunction:: cosinorage.DataHandlers.calculate_measure_time()


general utility functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.DataHandlers.filter_incomplete_days()

.. autofunction:: cosinorage.DataHandlers.filter_consecutive_days()

.. autofunction:: cosinorage.DataHandlers.largest_consecutive_sequence()

.. autofunction:: cosinorage.DataHandlers.calculate_enmo()

.. autofunction:: cosinorage.DataHandlers.calculate_minute_level_enmo()


visualization functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.DataHandlers.plot_orig_enmo()

.. autofunction:: cosinorage.DataHandlers.plot_enmo()

.. autofunction:: cosinorage.DataHandlers.plot_orig_enmo_freq()
   