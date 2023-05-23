"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable



class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.
        
        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points. 


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        # we will start with bisection to find initial approximations and then we will shift to newton
        # we will start with bisection to find initial approximations and then we will shift to newton
        def f(x):
            return f1(x) - f2(x)

        n = 100  # number of points
        list_of_roots = []

        if a == b:
            if f(a) == 0:
                list_of_roots.append(a)
            return list_of_roots

        list_of_ranges = list(np.linspace(a, b, n))
        if len(list_of_ranges) == 0:
            return list_of_roots
        elif len(list_of_ranges) == 1:
            num = list_of_ranges[0]
            if f(num) == 0:
                list_of_roots.append(num)
            return list_of_roots

        def bisection(p1, p2):
            l = 0
            while l > 50 and abs(f(p2) - f(p1)) > 2 * maxerr:
                z = 0.5 * (p1 + p2)
                if f(p1) * f(z) < 0:
                    p2 = z
                else:
                    p1 = z
                l += 1
            return (0.5 * (p1 + p2))

        # newton-raphson
        h = 0.00000001

        def dfx(x):
            return (f(x + h) - f(x)) / h

        for p in range(n - 1):
            curr_point = list_of_ranges[p]
            next_point = list_of_ranges[p + 1]
            if f(curr_point) == 0:  # if f(x)=0, x is a root and we add it to the answers
                list_of_roots.append(curr_point)
                continue
            elif f(curr_point) * f(next_point) > 0:  # if the result is positive there isn't a root in this range
                continue
            else:
                l = 0
                while abs(f(curr_point)) > maxerr:
                    curr_point = curr_point - f(curr_point) / dfx(curr_point)
                    l += 1
                    if curr_point < a or curr_point > b or l >= 50:
                        # if the point is out of range or the algorithm is stuck in a loop, try bisection
                        list_of_roots.append(bisection(list_of_ranges[p], list_of_ranges[p + 1]))
                        break
            list_of_roots.append(curr_point)

        return list_of_roots


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()
        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)
        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


if __name__ == "__main__":
    unittest.main()
