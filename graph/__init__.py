from .src.graph import Graph
import logging


logger = logging.getLogger()
handler = logging.StreamHandler()
# handler = logging.FileHandler('logger_output_file.txt')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.ERROR)
