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

    average_corr = []
    Ntrials = 10
    for i in range(Ntrials):
        print "Trial {} of {}".format(i+1, Ntrials)
        cluster_pts = utils.cluster(points, K=K, origin=origin, \
            distance=distance, compare=compare, combine=combine)
        average_corr.append(verify(correct_pts, cluster_pts, compare))

    return sum(average_corr) / Ntrials

def test_2d_pts():
    points = [  ( 1,  1), \
                ( 1,  1), \
                ( 1,  1), \
                ( 3,  3), \
                ( 3,  3), \
                ( 3,  3), \
                ( 5,  5), \
                ( 5,  5), \
                ( 5,  5), \
                ( 7,  7), \
                ( 7,  7), \
                ( 7,  7)  ]
    
    correct_pts = [ ( 1,  1), \
                    ( 3,  3), \
                    ( 5,  5), \
                    ( 7,  7)  ]

    distance= lambda p1, p2: (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    compare = lambda p1, p2, op: op(p1[0], p2[0]) and op(p1[1], p2[1])
    combine = lambda p1, p2: ( (p1[0] + p2[0])/2, (p1[1] + p2[1])/2 )
    origin = (0, 0)
    K = 4

    average_corr = []
    Ntrials = 10
    for i in range(Ntrials):
        print "Trial {} of {}".format(i+1, Ntrials)
        cluster_pts = utils.cluster(points, K=K, origin=origin, \
            distance=distance, compare=compare, combine=combine)
        average_corr.append(verify(correct_pts, cluster_pts, compare))

    return sum(average_corr) / Ntrials

if __name__ == '__main__':
    results = lambda r, s: "{} {} ({}%)".format("PASS" if r > 0.95 else "FAIL", s, r*100)
    print results(test_1d_pts(), "Clustering 1D points")
    print results(test_2d_pts(), "Clustering 2D points")
