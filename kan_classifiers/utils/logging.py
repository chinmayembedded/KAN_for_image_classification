import logging, sys, csv

LOG_FORMAT = "[%(levelname)-8s][%(asctime)s][%(funcName)-25s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_logger(filename, name=None, force=False):
    logging.basicConfig(filename=filename, level=logging.INFO,
                        format=LOG_FORMAT, datefmt=DATE_FORMAT, force=force)
    return logging.getLogger(name=name)


class CSVLogger:

    def __init__(self, filename:str, header_fields:dict):
        self.filename = filename
        self.header_fields = header_fields
        with open(self.filename, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.header_fields)
            writer.writeheader()

    def log(self, data:dict):
        with open(self.filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.header_fields)
            writer.writerow(data)