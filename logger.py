import logging

formatter = logging.Formatter('%(message)s')


def create_logger(name, filename, mode='a'):
    fh = logging.FileHandler(filename=filename, mode=mode)
    fh.setFormatter(formatter)

    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)
    log.addHandler(fh)

    return log
