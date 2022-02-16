import numpy
v = numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print("Original array:")
print(v)
sv = 1 / (1 + numpy.exp(-v))
print("numpy sigmod new array:")
print(numpy.around(sv, 6))
