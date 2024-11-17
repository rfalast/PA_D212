
class BaseModel:

    # constants
    NONE_VALUE = "NONE"

    # init method
    def __init__(self):
        pass

    # helper function to ensure that the value doesn't return none
    # when attempting to work with a string.
    def not_none(self, value_to_check):
        the_result = self.NONE_VALUE

        # make sure we're not none
        if value_to_check is None:
            the_result = self.NONE_VALUE
        elif not isinstance(value_to_check, str):
            the_result = str(value_to_check)
        else:
            the_result = value_to_check

        # return the result
        return the_result

    # check if the value is valid
    def is_valid(self, the_value, the_type=type):
        # default to False
        the_result = False

        # check if NONE
        if the_value is not None:

            # perform secondary check
            if the_type is not None:

                # optional check of isinstance
                if isinstance(the_value, the_type):
                    # set to True
                    the_result = True
                else:
                    # set to False
                    the_result = False

        # return the result
        return the_result
