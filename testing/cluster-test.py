#!/usr/bin/python
import sys
sys.path.append('../prototype')
import utils
import operator as op

verify = lambda ls1, ls2, compare: sum( map(lambda (p1, p2): compare(p1, p2, op.eq), zip(ls1, ls2))) / float(len(ls1))
def test_1d_pts():
    points = [1, 1, 1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 7, 7, 7, 7]
    correct_pts = [1, 3, 5, 7]
    distance= lambda p1, p2: p1 - p2
    compare = lambda p1, p2, op: op(p1, p2)
    combine = lambda p1, p2: (p1 + p2) / 2
    origin = 0
    K = 4

    cluster_pts = utils.cluster(points, K=K, origin=origin, \
        distance=distance, combine=combine)
    return verify(correct_pts, cluster_pts, compare)

def test_2d_pts():
    points = [  ( 1,  1), \
                ( 1,  1), \
                ( 1,  1), \
                ( 3,  3), \
                ( 3,  3), \
                ( 3,  3), \
                ( 3,  5), \
                ( 3,  5), \
                ( 5,  3), \
                ( 5,  3), \
                ( 5,  5), \
                ( 5,  5), \
                ( 5,  5), \
                ( 7,  7), \
                ( 7,  7), \
                ( 7,  7)  ]
    
    correct_pts = [ ( 1,  1), \
                    ( 3,  3), \
                    ( 4,  4), \
                    ( 5,  5), \
                    ( 7,  7)  ]

    distance= lambda (p1x, p1y), (p2x, p2y): (p1x - p2x)**2 + (p1y - p2y)**2
    compare = lambda (p1x, p1y), (p2x, p2y), op: op(op(p1x, p2x), op(p1y, p2y))
    combine = lambda (p1x, p1y), (p2x, p2y): ( (p1x + p2x)/2, (p1y + p2y)/2 )
    origin = (0, 0)
    K = 5
    print combine((3, 5), (5, 3))

    cluster_pts = utils.cluster(points, K=K, origin=origin, \
        distance=distance, combine=combine)
    return verify(correct_pts, cluster_pts, compare)

if __name__ == '__main__':
    results = lambda r, s: "{} {} ({}%)".format("PASS" if r > 0.95 else "FAIL", s, r*100)
    print results(test_1d_pts(), "Clustering 1D points")
    print results(test_2d_pts(), "Clustering 2D points")
