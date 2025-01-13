import logging

from model.analysis.models.ModelBase import ModelBase
from model.constants.BasicConstants import MT_KNN_CLASSIFICATION


# KNN Model tools
class KNN_Model(ModelBase):

    # init method
    def __init__(self, dataset_analyzer):
        # call superclass
        super().__init__(dataset_analyzer)

        # initialize logger
        self.logger = logging.getLogger(__name__)

        # log that we've been instantiated
        self.logger.debug("An instance of KNN_Model has been created.")

        # define the model type
        self.model_type = MT_KNN_CLASSIFICATION

        # force call to encode_dataframe() with drop_first=False
        self.variable_encoder.encode_dataframe(drop_first=False)

        # reload the encoded_df with drop_first set to False
        self.encoded_df = self.variable_encoder.get_encoded_dataframe(drop_first=False)

