import sympy as sym
from sympy import log, sin, pi
from sympy.parsing import sympy_parser
from sympy.parsing.mathematica import mathematica
import numpy as np
from math import pi, e, sqrt
import random
from mpl_toolkits import mplot3d
import base64
from io import BytesIO
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

X_1, X_2, X_3, X_4, X_5, X_6 = sym.symbols('X_1 X_2 X_3 X_4 X_5 X_6')
Pi = sym.symbols('Pi')
x_vector = [X_1, X_2, X_3, X_4, X_5, X_6]
FUNCTION_STRINGS = ["X_1**4+X_2**4-0.62*X_2**2-0.62*X_1**2", #suggested functions
                "100*(X_2-X_1**2)**2+(1-X_1)**2",
              "(X_1-X_2+X_3)**2 + (-X_1+X_2+X_3)**2+(X_1+X_2-X_3)**2",
              "(1+(X_1+X_2+1)**2*(19-14*X_1+3*X_1**2-14*X_2+6*X_1*X_2+3*X_2**2))*(30+(2*X_1-3*X_2)**2*(18-32*X_1+12*X_1**2+48*X_2-36*X_1*X_2+27*X_2**2))",
               "Exp[-21*Log[2]*(X_1-0.08)**2/(0.854**2)]*Sin[5*Pi*(X_1**(3/4)-0.05)]",
              "(X_1**2+X_2-11)**2+(X_1+X_2**2-7)**2-200"]

POINTS = [[1, 1], #suggested started points
          [-1.2, 1.0],
          [100.0, -1.0, 2.5],
          [0.4, -0.6],
          [random.random()],
          [-5, 5],]

GOLDEN_VALUE = (1+sqrt(5))/2 #Golden value

class Steepest_descent():
    """
    :param
    equation - input function in R^n, on which the Steepest Descent algorithm will work
    starting point - coordinates of each "X" - default starting point is vector of zeors
    function accuracy - the diffrence between equation value in two next iterations
    x_accuracy - the diffrence betweem coordinates in two next iterations
    gradient_accuracy - minimal value of gradient that we are working for (vector length)
    number_of_iterations - maximum number of iterations that our algoritm will go  through

    :returns
    None, it's instance of class.
    """

    def __init__(self, equation, starting_point=0, function_accuracy=1e-3, x_accuracy=1e-3, gradient_accuracy=1e-3, number_of_iterations=5000):

        self.function = self.parse_equation(equation)
        self.point = np.array(starting_point) if starting_point else np.zeros((len(self.function.free_symbols), 1))
        self.grad = np.array([sym.diff(self.function, x) for x in x_vector if x in self.function.free_symbols])

        self.path_x = [self.point[0]]
        self.path_y = [self.point[-1]]
        self.path_z = [np.float(self.calculate_function_value_in_point(self.function, self.point))]

        self.function_accuracy = float(function_accuracy)
        self.x_accuracy = float(x_accuracy)
        self.gradient_accuracy = float(gradient_accuracy)
        self.max_iter = int(number_of_iterations)

    def parse_equation(self, equation):
        """
        :param equation: equation from string input, to be converted
        :return: equation in form understand by Sympy library
        """
        equation.replace("^", "**")
        return mathematica(equation)

    def calculate_function_value_in_point(self, function, point):
        """
        :param function: any equation with vector of undefined variables
        :param point: coordinates to fill the functions with values
        :return: value of the function in point
        """
        return function.subs([(x, value) for x, value in zip(x_vector, point)])

    def get_vector_length(self, vector):
        return np.sqrt(np.float((vector.dot(vector))))

    def get_direction(self):
        return np.array([self.calculate_function_value_in_point(self.grad[i], self.point) for i in range(len(self.grad))])

    def get_best_step_distance(self, direction, prev_val=1000, lower_bound=0, upper_bound=10, depth=30, epsilon=1e-4):
        max_step = upper_bound - (GOLDEN_VALUE-1)*(upper_bound-lower_bound)
        min_step = lower_bound + (GOLDEN_VALUE-1)*(upper_bound-lower_bound)
        best_step_distance = max_step

        f_val_1 = self.calculate_function_value_in_point(self.function, self.point-max_step*direction)
        f_val_2 = self.calculate_function_value_in_point(self.function, self.point-min_step*direction)
        if f_val_2 < f_val_1:
            if(max_step > min_step):
                upper_bound = max_step
            else:
                lower_bound = max_step
        else:
            if min_step > max_step:
                upper_bound = min_step
            else:
                lower_bound = max_step
        if abs(prev_val-best_step_distance) <= epsilon or depth < 0:
            return best_step_distance
        else:
            return self.get_best_step_distance(direction, best_step_distance, lower_bound, upper_bound, depth-1, epsilon)

    def find_minimum(self):
        direction = self.get_direction()
        inf_loop_guardian = 0
        direction_magnitude = self.get_vector_length(direction)
        while direction_magnitude > self.gradient_accuracy and inf_loop_guardian < self.max_iter:
            step_distance = self.get_best_step_distance(direction, direction_magnitude)
            self.point = self.point - direction*step_distance
            if len(self.point) == 2:
                self.path_x.append(self.point[0])
                self.path_y.append(self.point[-1])
                self.path_z.append(np.float(self.calculate_function_value_in_point(self.function, self.point)))
            direction = self.get_direction()
            direction_magnitude = self.get_vector_length(direction)
            inf_loop_guardian=inf_loop_guardian+1
        print(direction_magnitude)
        return self.point

    def generate_plot(self, plot_size=6):
        if len(self.point) == 2:
            My_X, My_Y = np.meshgrid(range(10 * plot_size), range(10 * plot_size))
            My_X = (My_X - 5 * plot_size) / 5
            My_Y = (My_Y - 5 * plot_size) / 5
            My_Z = np.ones((10 * plot_size, 10 * plot_size))
            for i in range(10 * plot_size):
                for j in range(10 * plot_size):
                    My_Z[i, j] = self.calculate_function_value_in_point(self.function, [My_X[i, j], My_Y[i][0]])

            fig = Figure()
            ax = Axes3D(fig)
            ax.plot_wireframe(My_X, My_Y, My_Z)
            ax.plot3D(np.array(self.path_x), np.array(self.path_y), np.array(self.path_z), c='r', marker='o')
            #ax.scatter(np.array(self.path_x), np.array(self.path_y), np.array(self.path_z), c='r', marker='o')
            # Save it to a temporary buffer.
            buf = BytesIO()
            fig.savefig(buf, format="png")

            img = base64.b64encode(buf.getbuffer()).decode("ascii")
            return img
            #return plt
            #plt.show()
        else:
            print("This is not 3D object to plot")


#a = Steepest_descent(FUNCTION_STRINGS[0], POINTS[0])
#a.generate_plot(6)
