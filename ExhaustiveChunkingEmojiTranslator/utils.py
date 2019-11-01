from numpy import dot
from numpy.linalg import norm


def cosine(a, b):
    return 1 - (dot(a, b) / (norm(a) * norm(b)))
