import logging
import os
from  datetime import datetime



log_file_name = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
log_path      = os.path.join(os.getcwd(), "logs", datetime.now().strftime('%d_%m_%Y'))
os.makedirs(log_path, exist_ok=True)
log_file_name = os.path.join(log_path, log_file_name)

logging.basicConfig(filename=log_file_name, level=logging.INFO, format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s")

# if __name__ == "__main__":
#     logging.info("Testing logging data")