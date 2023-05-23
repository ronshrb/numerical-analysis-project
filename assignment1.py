"""
In this assignment you should interpolate the given function.
"""

import numpy as np
import time
import random



class Assignment1:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        Interpolate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the interpolation error.
        Your secondary objective is minimizing the running time. 
        The assignment will be tested on variety of different functions with 
        large n values. 
        
        Interpolation error will be measured as the average absolute error at 
        2*n random points between a and b. See test_with_poly() below. 

        Note: It is forbidden to call f more than n times. 

        Note: This assignment can be solved trivially with running time O(n^2)
        or it can be solved with running time of O(n) with some preprocessing.
        **Accurate O(n) solutions will receive higher grades.** 
        
        Note: sometimes you can get very accurate solutions with only few points, 
        significantly less than n. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """
        #

        x_values = np.linspace(a,b,n)  # taking x values within the range
        y_values = [f(num) for num in x_values]  # taking y values using our x values

        def thomas(w, k):
            """
            thomas algorithm
            Parameters
            ----------
            w : coefficients matrix
            k : points matrix

            Returns
            -------
            a matrix of control points for the bezier curves."""
            n = len(k)
            c = list(np.zeros(n))
            # for i in range(n - 1):
            #     w[i + 1] = [w[i + 1][j] - (w[i + 1][i] / w[i][i]) * w[i][j] for j in range(n)]
            #     k[i + 1] = k[i + 1] - (w[i + 1][i] / w[i][i]) * k[i]
            # c[n - 1] = k[n - 1] / w[n - 1][n - 1]
            # for i in range(n - 2, -1, -1):
            #     c[i] = (k[i] - sum([w[i][j] * c[j] for j in range(i + 1, n)])) / w[i][i]
            for i in range(n - 1):
                factor = w[i][i]
                for j in range(i + 1, n):
                    ratio = w[j][i] / factor
                    for l in range(i + 1, n):
                        w[j][l] -= ratio * w[i][l]
                    k[j] -= ratio * k[i]

            for i in range(n - 1, -1, -1):
                factor = w[i][i]
                c[i] = k[i]
                for j in range(i + 1, n):
                    c[i] -= w[i][j] * c[j]
                c[i] /= factor

            return c

        def coeffs(points):
            """
            a function that builds the w matrix and the k matrix
            :param points: the given function's y values
            :return: A, B: control points for the curves
            """
            n = len(points) - 1
            w = 4 * np.identity(n)
            np.fill_diagonal(w[1:], 1)
            np.fill_diagonal(w[:, 1:], 1)
            w[0, 0] = 2
            w[n - 1, n - 1] = 7
            w[n - 1, n - 2] = 2
            k = [2 * (2 * points[i] + points[i + 1]) for i in range(n)]
            k[0] = points[0] + 2 * points[1]
            k[n - 1] = 8 * points[n - 1] + points[n]
            A = thomas(w, k) # we find A using thomas algorithm
            B = [0] * n
            for i in range(n - 1):
                B[i] = 2 * points[i + 1] - A[i + 1]
            B[n - 1] = (A[n - 1] + points[n]) / 2
            return A, B

        def get_curve(x1, a, b, x2):
            """
            :param x1:  first point of the curve
            :param a: control point from A
            :param b: control point from B
            :param x2: last point of the curve
            :return: a function of the curve using t
            """
            def curve_func(t):
                return np.power(1 - t, 3) * x1 + 3 * a * t * np.power(1 - t, 2) + 3 * b * (1 - t) * np.power(t,2)  + x2 * np.power(t, 3)
            return curve_func

        ac, bc = coeffs(y_values)    # we get the control points for each curve by using y values
        def poly(x):
            """
            finding the right curve using the x we are given
            :param x: int, an x value
            :return: the y value of the curve that x value is on
            """
            n = len(x_values)-1
            idx = 0
            for i in range(n):      # we find the location of the x value in the points we sampled
                if x_values[i] <= x <= x_values[i+1]:
                    idx = i
                    break
            t = (x - a) / (b - a)       # we find the t value that matches with x
            curr_curve = get_curve(y_values[idx], ac[idx], bc[idx], y_values[idx + 1])
            result = curr_curve(t *n - idx)
            return result

        return poly


##########################################################################


import unittest
from functionUtils import *
from tqdm import tqdm


class TestAssignment1(unittest.TestCase):

    def test_with_poly(self):
        T = time.time()

        ass1 = Assignment1()
        mean_err = 0

        d = 30
        for i in tqdm(range(100)):
            a = np.random.randn(d)

            f = np.poly1d(a)

            ff = ass1.interpolate(f, -10, 10, 100)

            xs = np.random.random(200)
            err = 0
            for x in xs:
                yy = ff(x)
                y = f(x)
                err += abs(y - yy)

            err = err / 200
            mean_err += err
        mean_err = mean_err / 100

        T = time.time() - T
        print(T)
        print(mean_err)

    def test_with_poly_restrict(self):
        ass1 = Assignment1()
        a = np.random.randn(5)
        f = RESTRICT_INVOCATIONS(10)(np.poly1d(a))
        ff = ass1.interpolate(f, -10, 10, 10)
        xs = np.random.random(20)
        for x in xs:
            yy = ff(x)

if __name__ == "__main__":
    unittest.main()
