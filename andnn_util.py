from time import time as current_time
from sys import stdout
import cv2 as cv


def is_image(filename):
    x = cv.imread(filename)
    try:
        x.shape
    except AttributeError:
        return False
    return True


class Timer:
    """A simple tool for timing code while keeping it pretty."""
    def __init__(self, mes='', pretty_time=True):
        self.mes = mes  # append after `mes` + '...'
        self.pretty_time = pretty_time

    @staticmethod
    def format_time(et):
        if et < 60:
            return '{:.1f} sec'.format(et)
        elif et < 3600:
            return '{:.1f} min'.format(et/60)
        else:
            return '{:.1f} hrs'.format(et/3600)

    def __enter__(self):
        stdout.write(self.mes + '...')
        stdout.flush()
        self.t0 = current_time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.t1 = current_time()
        if self.pretty_time:
            print("done (in {})".format(self.format_time(self.t1 - self.t0)))
        else:
            print("done (in {} seconds).".format(self.t1 - self.t0))
        stdout.flush()



