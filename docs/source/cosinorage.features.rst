cosinorage.features module
===========================

module contents
---------------

.. automodule:: cosinorage.features
   :members:
   :undoc-members:
   :show-inheritance:

classes
----------------------------------------

.. autoclass:: cosinorage.features.WearableFeatures
   :members:

utility functions
----------------------------------------

cosinor (circadian rhythm analysis) analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.cosinor_by_day()

.. autofunction:: cosinorage.features.cosinor_multiday()


nonparametric (circadian rhythm analysis) analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.IV()

.. autofunction:: cosinorage.features.IS()

.. autofunction:: cosinorage.features.RA()

.. autofunction:: cosinorage.features.M10()

.. autofunction:: cosinorage.features.L5()

physical activity metrics
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.activity_metrics()

sleep metrics
~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.apply_sleep_wake_predictions()

.. autofunction:: cosinorage.features.waso()

.. autofunction:: cosinorage.features.tst()

.. autofunction:: cosinorage.features.pta()

.. autofunction:: cosinorage.features.sri()


rescaling functions
~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.utils.rescaling.min_max_scaling_exclude_outliers()


visualization functions
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: cosinorage.features.utils.visualization.plot_sleep_predictions()

.. autofunction:: cosinorage.features.utils.visualization.plot_non_wear()

.. autofunction:: cosinorage.features.utils.visualization.plot_cosinor()

