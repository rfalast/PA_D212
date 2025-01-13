import logging

from util.BaseUtil import BaseUtil


# class holding common utility functions.
class Model_Result_Populator(BaseUtil):

    # init method
    def __init__(self):
        super().__init__()

        # initialize logger
        self.logger = logging.getLogger(__name__)

        # variable declaration
        self.storage = {}

        # log that we've been instantiated
        self.logger.debug("An instance of Model_Result_Populator has been created.")

    # populate storage with key
    def populate_storage(self, the_key: str, the_item):
        # validate arguments
        if not isinstance(the_key, str):
            raise AttributeError("the_key is None.")
        elif the_item is None:
            raise AttributeError("the_item is None.")

        # log what we're doing.
        self.logger.debug(f"Adding [type({the_item})] to key [{the_key}].")

        # add to storage
        self.storage[the_key] = the_item

    # remove from storage.  If you pass a key that is not present, nothing will happen.
    def remove_from_storage(self, the_key: str):
        # validate arguments
        if not isinstance(the_key, str):
            raise AttributeError("the_key is None.")

        # log the request, not what we actually do.
        self.logger.debug(f"A request to remove item stored under key [{the_key}].")

        # check if the key is in storage
        if the_key in self.storage.keys():
            # log what we're doing
            self.logger.debug(f"removing [{the_key}] from storage.")

            # remove key from storage
            del self.storage[the_key]
        # key is not in storage
        else:
            # log that we did nothing.
            self.logger.debug(f"[{the_key}] is not present in storage.")

    # get storage
    def get_storage(self) -> dict:
        return self.storage

