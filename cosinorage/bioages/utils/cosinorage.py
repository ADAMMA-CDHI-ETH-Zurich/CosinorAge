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

from typing import List
from datahandlers.datahandler import DataHandler

# TODO:
# - do we need the chronoAges (read them as a list)?
# - do we need the model params

class CosinorAge():
    def __init__(self,
        handlers: List[DataHandler],
        model_params: dict):

        self.handlers = handlers
        self.model_params = model_params
        
        self.cosinorAges = []

        self.__predict()

    def __predict(self):
        pass