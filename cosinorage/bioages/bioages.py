###########################################################################
# Copyright (C) 2024 ETH Zurich
# CosinorAge: Prediction of biological age based on accelerometer data
# using the CosinorAge method proposed by Shim, Fleisch and Barata
# (https://www.nature.com/articles/s41746-024-01111-x)
# 
# Authors: Jacob Leo Oskar Hunecke
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#         http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##########################################################################

from ..datahandlers import DataHandler
from ..features.utils.cosinor_analysis import cosinor_multiday

import numpy as np
from typing import List

# model parameters
model_params_generic = {
    "shape": 0.01462774,
    "rate": -13.36715309,
    "mesor": -0.03204933,
    "amp1": -0.01971357,
    "phi1": -0.01664718,
    "age": 0.10033692
}

model_params_female = {
    "shape": 0.01294402,
    "rate": -13.28530410,
    "mesor": -0.02569062,
    "amp1": -0.02170987,
    "phi1": -0.13191562,
    "age": 0.08840283
}

model_params_male = {
    "shape": 0.013878454,
    "rate": -13.016951633,
    "mesor": -0.023988922,
    "amp1": -0.030620390,
    "phi1": 0.008960155,
    "age": 0.101726103
}

m_n = -1.405276
m_d = 0.01462774
BA_n = -0.01447851
BA_d = 0.112165
BA_i = 133.5989


class CosinorAge:
    def __init__(self, records: List[dict]):
        self.records = records
        
        self.model_params_generic = model_params_generic
        self.model_params_female = model_params_female
        self.model_params_male = model_params_male

        self.__compute_cosinor_ages()

    def __compute_cosinor_ages(self):
        for record in self.records:
            result = cosinor_multiday(record["handler"].get_ml_data())[0]
            record["amplitude"] = result["amplitude"]*1000
            record["acrophase"] = result["acrophase_time"]
            record["mesor"] = result["MESOR"]*1000
            
            gender = record.get("gender", "unknown")
            if gender == "female":
                coef = self.model_params_female
            elif gender == "male":
                coef = self.model_params_male
            else:
                coef = self.model_params_generic

            m = coef["mesor"] * record["mesor"]
            a = coef["amp1"] * record["amplitude"]
            p = coef["phi1"] * record["acrophase"]
            b = coef["age"] * record["age"]
            n1 = m + a + p + b
            xb = n1 + coef["shape"]

            m_val = 1 - np.exp((m_n * np.exp(xb)) / m_d)

            # Calculate cosinor age
            record["cosinoage"] = float(((np.log(BA_n * np.log(1 - m_val))) / BA_d) + BA_i)

            # Calculate cosinor age advance
            record["cosinoage_advance"] = float(record["cosinoage"] - record["age"])

    def get_predictions(self):
        return self.records




