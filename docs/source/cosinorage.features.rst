cosinorage.features Module
==========================

Module Contents
---------------

.. automodule:: cosinorage.features
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

.. autoclass:: cosinorage.features.WearableFeatures
   :members:

Utility Functions
-----------------

Cosinor (Circadian Rhythm Analysis) Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.cosinor_by_day()

.. autofunction:: cosinorage.features.cosinor_multiday()


Non-parametric (Circadian Rhythm Analysis) Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.IV()

.. autofunction:: cosinorage.features.IS()

.. autofunction:: cosinorage.features.RA()

.. autofunction:: cosinorage.features.M10()

.. autofunction:: cosinorage.features.L5()

Physical Activity Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.activity_metrics()

Sleep Metrics
~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.apply_sleep_wake_predictions()

.. autofunction:: cosinorage.features.waso()

.. autofunction:: cosinorage.features.tst()

.. autofunction:: cosinorage.features.pta()

.. autofunction:: cosinorage.features.sri()


Rescaling Functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.utils.rescaling.min_max_scaling_exclude_outliers()


Visualization Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.utils.visualization.plot_sleep_predictions()

.. autofunction:: cosinorage.features.utils.visualization.plot_non_wear()

.. autofunction:: cosinorage.features.utils.visualization.plot_cosinor()

