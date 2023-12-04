import sys
from src.logger import logging

def get_error_details(error_message, sys_detail):
    _,_,traceback = sys_detail.exc_info()
    file_name       = traceback.tb_frame.f_code.co_filename
    error_message   = "Error details: Script Name: [{0}] Line No: [{1}] Error Message: [{2}]".format(file_name, traceback.tb_lineno, str(error_message))
    return error_message

class CustomException(Exception):

    def __init__(self, error_message, sys_detail):
        super().__init__(error_message)
        self.error_message = get_error_details(error_message, sys_detail)

    def __str__(self):
        return repr(self.error_message)

if __name__ == "__main__":
    try:
        a = 2/0
    except Exception as e:
        logging.info(f"Logging details: {e}")
        raise CustomException(e, sys)

