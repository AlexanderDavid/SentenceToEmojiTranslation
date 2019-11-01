from numpy import dot
from numpy.linalg import norm


def cosine(a, b):
    return 1 - (dot(a[0], b[0]) / (norm(a[0]) * norm(b[0])))
