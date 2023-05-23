"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import time
import random
import assignment2


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        # a, b = np.float32(a), np.float32(b)
        if n == 1:      # if there is only one point, use mid-point rule
            h = (np.float32(b) - np.float32(a)) / n
            result = abs(a + h/2)
            return np.float32(result)
        if n == 2:      # if there is only 2 points, use trapezoidal rule
            h = (np.float32(b) - np.float32(a)) / n
            result = abs((f(a) + 2*(a+h)+f(b))*h/2)
            return np.float32(result)
        if n%2 != 0:    # checks if n is odd, if it is we lower it by 1
            n -= 1
        else:
            n -= 2      # checks if n is even, if it is we lower it by 2
        # composite simpson
        h = (np.float32(b) - np.float32(a)) / n
        x = a
        total = 0
        s = f(a) + f(b)
        for i in range(n-1):
            x+=h
            y = f(x)
            if i % 2 == 0:
                total += np.float32(4*y)
            else:
                total += np.float32(2*y)
        result = abs((h/3)*(s+total))
        return np.float32(result)


    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        def f(x):
            return f1(x)-f2(x)
        ass2 = assignment2.Assignment2()
        ranges = []
        for i in range(1,100,9):
            # we find the intersections between the functions so they would be used as ranges
            ranges += ass2.intersections(f1,f2,a=i,b=i+9,maxerr=0.001)
        ranges = sorted(list(set(ranges)))
        area = np.float32(0)
        n = len(ranges)
        if n < 2:
            # if there is less than two points in the list, there isn't an area between the functions
            return np.nan
        for i in range(0,n-1):
            # for each two points i calculate the area between them and sum it to the rest
            area += abs(self.integrate(f, ranges[i], ranges[i+1], 100))

        return np.float32(area)


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment3(unittest.TestCase):

    def test_integrate_float32(self):
        ass3 = Assignment3()
        f1 = np.poly1d([-1, 0, 1])
        r = ass3.integrate(f1, -1, 1, 10)

        self.assertEqual(r.dtype, np.float32)

    def test_integrate_hard_case(self):
        ass3 = Assignment3()
        f1 = strong_oscilations()
        r = ass3.integrate(f1, 0.09, 10, 20)
        true_result = -7.78662 * 10 ** 33
        self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))


if __name__ == "__main__":
    unittest.main()
