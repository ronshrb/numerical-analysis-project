"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""
import math

import numpy as np
import time
import random
from functionUtils import AbstractShape
from sklearn.cluster import KMeans



class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, contour):
        # super(MyShape, self).__init__()
        self.contour = contour

    def area(self):
        result = np.float32(0)
        points = self.contour
        n = len(points)
        j = n - 1
        for i in range(n):      # shoelace method
            result += np.float32(points[j][0] + points[i][0]) * np.float32(points[j][1] - points[i][1])
            j = i
        return np.float32(abs(result) / 2.0)
        # points = self.contour
        # n = len(points)
        # total = 0.0
        # for i in range(n):
        #     j = (i + 1) % n
        #     total += points[i][0] * points[j][1]
        #     total -= points[j][0] * points[i][1]
        # result = abs(total) / 2
        # return np.float32(result)


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def area(self, contour: callable, maxerr=0.001)->np.float32:
        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        # ndots = int(10000 * (0.9 - (maxerr / 100)))
        ndots = 330
        area = np.float32(0)
        points = contour(ndots)     # we sample the points of the shape
        n = len(points)
        j = n - 1
        for i in range(n):      # we use the shoelace method to calculate the area
            area += np.float32(points[j][0] + points[i][0]) * np.float32(points[j][1] - points[i][1])
            j = i
        return np.float32(abs(area) / 2.0)

        # def ndots(x):
        #     return int(10000 * (1 - (x * 100 / 10000)))
        # area = np.float32(0)
        # points = contour(ndots(maxerr))
        # j = len(points) - 1
        # for i in range(0, len(points)):
        #     # shoelace area
        #     area += abs((points[j][0] + points[i][0]) * (points[j][1] - points[i][1]))
        #     j = i
        # area = area / 2
        # return np.float32(area)

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable.
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        An object extending AbstractShape.
        """
        # result = None
        # replace these lines with your solution
        start_time = time.time()
        timelimit = 5+maxtime*0.8

        # Define the model function
        # def model_func(x, a, b, c):
        #     return a * np.sin(b * x + c)

        # Sample the data points
        data = []
        n = 40000
        for i in range(n):
            # time1 = time.time() - start_time
            # if the running time succeeded the timeout we set, break out of the loop
            if time.time() - start_time > timelimit:
                break
        # while time.time() - start_time < timeout:
        #     time1 = time.time() - start_time
            # creating the list of points we sample
            x, y = sample()
            data.append([x, y])
        clus = int(1+len(data)/1000)    # a number of clusters for kmeans
        # n = len(data)
        # if n < 1:
        #     return True
        kmeans = KMeans(n_clusters=clus,n_init=1).fit(data)

        # fitting the points we sampled
        fitted_points = kmeans.cluster_centers_
        fitted_points = np.array(fitted_points)

        # finding the center point of the shape
        center = np.mean(fitted_points, axis=0)

        # finding the angles of each point in our fitted points list by using the center point
        angles = np.arctan2(fitted_points[:, 1] - center[1], fitted_points[:, 0] - center[0])

        # sorting the points by their angles so we could get the points in a clockwise order
        sorted_points = fitted_points[np.argsort(angles)]
        result = MyShape(sorted_points)
        return result


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
