cosinorage.dataloaders module
=============================

module contents
---------------

.. automodule:: cosinorage.dataloaders
   :no-members:


classes
----------------------------------------

.. autoclass:: cosinorage.dataloaders.DataLoader
   :members:

.. autoclass:: cosinorage.dataloaders.NHANESDataLoader
   :members:

.. autoclass:: cosinorage.dataloaders.GalaxyDataLoader
   :members:

.. autoclass:: cosinorage.dataloaders.UKBDataLoader
   :members:

utility functions
----------------------------------------

galaxy smartwatch data functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.dataloaders.read_galaxy_data()

.. autofunction:: cosinorage.dataloaders.filter_galaxy_data()

.. autofunction:: cosinorage.dataloaders.resample_galaxy_data()

.. autofunction:: cosinorage.dataloaders.preprocess_galaxy_data()

.. autofunction:: cosinorage.dataloaders.acceleration_data_to_dataframe()

.. autofunction:: cosinorage.dataloaders.calibrate()

.. autofunction:: cosinorage.dataloaders.remove_noise()

.. autofunction:: cosinorage.dataloaders.detect_wear()

.. autofunction:: cosinorage.dataloaders.calc_weartime()

uk biobank data functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.dataloaders.read_ukb_data()

.. autofunction:: cosinorage.dataloaders.filter_ukb_data()

.. autofunction:: cosinorage.dataloaders.resample_ukb_data()


nhanes data functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.dataloaders.read_nhanes_data()

.. autofunction:: cosinorage.dataloaders.filter_nhanes_data()

.. autofunction:: cosinorage.dataloaders.resample_nhanes_data()

.. autofunction:: cosinorage.dataloaders.remove_bytes()

.. autofunction:: cosinorage.dataloaders.clean_data()

.. autofunction:: cosinorage.dataloaders.calculate_measure_time()


general utility functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.dataloaders.filter_incomplete_days()

.. autofunction:: cosinorage.dataloaders.filter_consecutive_days()

.. autofunction:: cosinorage.dataloaders.largest_consecutive_sequence()

.. autofunction:: cosinorage.dataloaders.calculate_enmo()

.. autofunction:: cosinorage.dataloaders.calculate_minute_level_enmo()


visualization functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.dataloaders.plot_orig_enmo()

.. autofunction:: cosinorage.dataloaders.plot_enmo()

.. autofunction:: cosinorage.dataloaders.plot_orig_enmo_freq()
   