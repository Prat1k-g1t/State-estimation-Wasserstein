# SPDX-FileCopyrightText: The sinkhorn-rom authors, see file AUTHORS
#
# SPDX-License-Identifier: GPL-3.0-or-later

from collections import defaultdict

from ..DataManipulators.DataStruct import QueryDataStruct, TargetDataStruct
from ...utils import timeit


class BaseModel:
    def __init__(self, name):
        self.name = name
        self.summary = defaultdict(list)

    def fit(self, query: QueryDataStruct, target: TargetDataStruct):
        """This method is specific to each model.
        """
        raise Exception('Not implemented.')

    def predict(self, query: QueryDataStruct, **kwargs):
        """
        """
        with timeit('Computing Prediction(s)', font_style='bold', bg='Blue', fg='White'):
            predictions = []
            for q in query:
                predictions.append(self.core_predict(q, **kwargs))
            return predictions

    def core_predict(self, query: QueryDataStruct, **kwargs):
        """This method is specific to each model.
        """
        raise Exception('Not implemented.')

    def refresh_summary(self):
        self.summary = defaultdict(list)

    # def predict_times(self, query: QueryDataStruct, **kwargs):
    #     """
    #     """
    #     with timeit('Computing Prediction(s)', font_style='bold', bg='Blue', fg='White'):
    #         predictions = []
    #         prediction_times = []
    #         for q in query:
    #             pred, pred_time = self.core_predict(q, **kwargs)
    #             predictions.append(pred)
    #             prediction_times.append(pred_time)
    #         return predictions, prediction_times

    # def knn_predict(self, query: QueryDataStruct,target:TargetDataStruct, **kwargs):
    #     """
    #     """
    #     with timeit('Computing Prediction(s)', font_style='bold', bg='Blue', fg='White'):
    #         predictions = []
    #         for (q,t) in  zip(query,target):
    #             predictions.append(self.knn_core_predict(q,t, **kwargs))
    #         return predictions

    # def knn_core_predict(self, query: QueryDataStruct,target:TargetDataStruct, **kwargs):
    #     """This method is specific to each model.
    #     """
    #     raise Exception('Not implemented.')


        
    def weights_core_predict(self, query: QueryDataStruct, target:TargetDataStruct, **kwargs):
        """This method is specific to each model.
        """
        raise Exception('Not implemented.')
    def weights_predict(self, query: QueryDataStruct, target:TargetDataStruct, **kwargs):
        """
        """
        with timeit('Computing Prediction(s)', font_style='bold', bg='Blue', fg='White'):
            results = [self.weights_core_predict(q,t,**kwargs) for (q,t) in zip(query,target)]
            return results

    # def embedding_core_predict(self, query: QueryDataStruct,target:TargetDataStruct, **kwargs):
    #     """This method is specific to each model.
    #     """
    #     raise Exception('Not implemented.')
    # def embedding_predict(self, query: QueryDataStruct,target:TargetDataStruct, **kwargs):
    #     """
    #     """
    #     with timeit('Computing Prediction(s)', font_style='bold', bg='Blue', fg='White'):
    #         #results = [self.embedding_core_predict(q, **kwargs) for q in query]
    #         results = [self.embedding_core_predict(q,t,**kwargs) for (q,t) in zip(query,target)]
    #         return results


