#!/usr/bin/python
import sys
sys.path.append('../prototype')
import utils
import operator as op

def test(points, value):
    K = len(points)
    flat_points = []
    for sl in points:
        flat_points.extend(sl)
    verify = lambda ls1, ls2: (len([x for x in ls1]) - len(list([x for x in ls1 if x not in ls2] + \
            [x for x in ls2 if x not in ls1])))/(len([x for x in ls1]))
    return verify(points, utils.cluster(flat_points, value=value, K=K))

def test_1d_pts():
    points = [[1, 1, 1, 1], \
              [3, 3, 3, 3], \
              [5, 5, 5, 5], \
              [7, 7, 7, 7]]
    value = lambda p: p
    return test(points, value)

def test_2d_pts():
    points = [[(1, 1), (1, 1), (1, 1)], \
              [(3, 3), (3, 3), (3, 3)], \
              [(3, 5), (3, 5), (5, 3), (5, 3)], \
              [(5, 5), (5, 5), (5, 5)], \
              [(7, 7), (7, 7), (7, 7)]]
    value = lambda (px, py): px**2 + py**2
    return test(points, value)

if __name__ == '__main__':
    results = lambda r, s: "{} {} ({}%)".format("PASS" if r > 0.95 else "FAIL", s, r*100)
    print results(test_1d_pts(), "Clustering 1D points")
    print results(test_2d_pts(), "Clustering 2D points")
