CosinorAge Package - Detailed
============================

The CosinorAge Python package offers a comprehensive, end-to-end solution for predicting the CosinorAge biomarker [9]. It seamlessly integrates all necessary steps, starting from data loading and preprocessing, to the computation of wearable features, and ultimately, the prediction of the biological age biomarker.

Package Overview
----------------
The package is organized into three main modules:

- **DataHandler**: Facilitates data loading and preprocessing. This module includes specialized subclasses designed to handle data from diverse sources, such as the Samsung Galaxy Smartwatch, NHANES, and UK Biobank, ensuring compatibility with a wide range of datasets.
- **WearableFeatures**: Extracts a comprehensive suite of wearable features from minute-level ENMO data. These include circadian rhythm metrics, sleep parameters, and physical activity features, offering a detailed analysis of health-related behaviors.
- **CosinorAge**: Predicts the CosinorAge biomarker by leveraging the processed minute-level ENMO data alongside the individual’s chronological age.

.. figure:: docs/figs/flowchart.png
   :alt: Supported Data Sources

   Modular architecture of the CosinorAge Python package.

DataHandler
-----------
The **DataHandler** class is a fundamental component of the package, ensuring that downstream modules (WearableFeatures and CosinorAge) receive the required data in the correct format—specifically, minute-level ENMO data. The following is a simplified class declaration:

.. code-block:: python

   import pandas as pd

   class DataHandler:
       def __init__(self):
           self.raw_data = pd.DataFrame()
           self.sf_data = pd.DataFrame()
           self.ml_data = pd.DataFrame()
           self.meta_dict = {}

       def __load_data(self):
           raise NotImplementedError("__load_data() should be implemented by subclasses")

The class declares the following members:

- **self.raw_data**: Stores raw data as read from source (e.g., accelerometer or ENMO data).
- **self.sf_data**: Stores preprocessed data in its original sampling frequency.
- **self.ml_data**: Stores minute-level ENMO data for downstream tasks.
- **self.meta_dict**: Stores metadata such as sampling frequency, data source, and units.

UKBDataHandler
--------------
The **UKBDataHandler** class is a subclass of DataHandler that specifically handles UK Biobank data.

.. code-block:: python

   class UKBDataHandler(DataHandler):
       def __init__(self, qa_file_path: str, ukb_file_dir: str, eid: int):
           super().__init__()
           self.qa_file_path = qa_file_path
           self.ukb_file_dir = ukb_file_dir
           self.eid = eid
           self.__load_data()

WearableFeatures
----------------
The **WearableFeatures** class computes a broad range of wearable-derived features from minute-level ENMO data collected over multiple days.

.. code-block:: python

   import pandas as pd

   class WearableFeatures():
       def __init__(self, handler: DataHandler, features_args: dict = {}):
           self.ml_data = handler.get_ml_data()
           self.feature_df = pd.DataFrame()
           self.feature_dict = {}
           self.__run()

Circadian Rhythm Analysis - Cosinor Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Cosinor analysis assesses the circadian rhythm using periodic data:

.. math::
   Y(t) = M + A \cos\left( \frac{2\pi t}{\tau} + \phi \right) + \epsilon(t)

where:

- **Y(t)**: Activity level (ENMO) at time t
- **M**: MESOR (Midline Estimating Statistic of Rhythm)
- **A**: Amplitude
- **φ**: Acrophase (peak activity time)
- **τ**: Period (typically 24h)
- **ϵ(t)**: Error term

.. figure:: docs/figs/cosinor_model.png
   :alt: Cosinor Model Example

   Cosinor model fitted to a 5-day ENMO dataset.

Physical Activity Metrics
-------------------------
Physical activity is classified into **sedentary, light, moderate, and vigorous** levels using predefined ENMO cutpoints.

Sleep Metrics
-------------
The package employs the **Cole-Kripke Algorithm** for sleep detection. The algorithm processes ENMO data and applies thresholding and rescoring rules to classify sleep and wake states.

CosinorAge
----------
The **CosinorAge** class predicts the CosinorAge biomarker based on wearable-derived features.

.. code-block:: python

   from typing import List

   class CosinorAge():
       def __init__(self, records: List[DataHandler]):
           self.handlers = handlers
           self.model_params = model_params
           self.cosinorAges = []
           self.__predict()

References
----------
[1] Roger J Cole, Daniel F Kripke, et al. Automatic sleep/wake identification from wrist activity. *Sleep*, 1992.
[2] Germaine Cornelissen. Cosinor-based rhythmometry. *Theoretical Biology and Medical Modelling*, 2014.
[3] David R Cox. Regression models and life-tables. *Journal of the Royal Statistical Society*, 1972.

For a complete list of references, please refer to the full documentation.
