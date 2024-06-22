import numpy as np
import matplotlib.pyplot as plt

class BarycentricInterpolation:
    def __init__(self, points, values, verbose=False):
        """
        Initialize with a set of points and corresponding function values.

        :param points: A list of tuples representing the (x1, x2) coordinates of points.
        :param values: A list of function values corresponding to the points.
        :param verbose: Boolean flag to control verbosity of the output.
        """
        self.points = np.array(points)
        self.values = np.array(values)
        self.verbose = verbose
    
    def compute_closest_points(self, y):
        """
        Compute the points A, B, C, and D around the point y.

        :param y: A tuple (y1, y2) representing the query point.
        :return: A, B, C, D points or None if any point is not found.
        """
        A = self._find_closest_point(y, lambda x: x[0] > y[0] and x[1] > y[1])
        B = self._find_closest_point(y, lambda x: x[0] > y[0] and x[1] < y[1])
        C = self._find_closest_point(y, lambda x: x[0] < y[0] and x[1] < y[1])
        D = self._find_closest_point(y, lambda x: x[0] < y[0] and x[1] > y[1])
        
        if self.verbose:
            print(f"Point y: {y}")
            print(f"Selected A: {A}")
            print(f"Selected B: {B}")
            print(f"Selected C: {C}")
            print(f"Selected D: {D}")

        if A is None or B is None or C is None or D is None:
            return None
        
        return A, B, C, D
    
    def _find_closest_point(self, y, condition):
        """
        Find the closest point to y satisfying the given condition.

        :param y: A tuple (y1, y2) representing the query point.
        :param condition: A lambda function to apply the condition.
        :return: The closest point satisfying the condition or None.
        """
        filtered_points = [point for point in self.points if condition(point)]
        if not filtered_points:
            return None
        distances = [np.linalg.norm(np.array(point) - np.array(y)) for point in filtered_points]
        return filtered_points[np.argmin(distances)]
    
    def is_inside_triangle(self, y, A, B, C):
        """
        Check if the point y is inside the triangle ABC.

        :param y: A tuple (y1, y2) representing the query point.
        :param A, B, C: Tuples representing the vertices of the triangle.
        :return: True if y is inside the triangle, else False.
        """
        r1, r2, r3 = self.compute_barycentric_coordinates(y, A, B, C)
        return 0 <= r1 <= 1 and 0 <= r2 <= 1 and 0 <= r3 <= 1

    def compute_barycentric_coordinates(self, y, A, B, C):
        """
        Compute the barycentric coordinates of y with respect to triangle ABC.

        :param y: A tuple (y1, y2) representing the query point.
        :param A, B, C: Tuples representing the vertices of the triangle.
        :return: Barycentric coordinates r1, r2, r3.
        """
        denom = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r1 = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denom
        r2 = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denom
        r3 = 1 - r1 - r2
        return r1, r2, r3

    def interpolate(self, y):
        """
        Interpolate the function value at point y using barycentric interpolation.

        :param y: A tuple (y1, y2) representing the query point.
        :return: Interpolated function value or NaN if not possible.
        """
        closest_points = self.compute_closest_points(y)
        if closest_points is None:
            if self.verbose:
                print(f"Can't compute with given sample for point {y}")
            return float('NaN')
        A, B, C, D = closest_points
        
        if self.is_inside_triangle(y, A, B, C):
            r1, r2, r3 = self.compute_barycentric_coordinates(y, A, B, C)
            return r1 * self.values[self._find_index(A)] + r2 * self.values[self._find_index(B)] + r3 * self.values[self._find_index(C)]
        
        if self.is_inside_triangle(y, C, D, A):
            r1, r2, r3 = self.compute_barycentric_coordinates(y, C, D, A)
            return r1 * self.values[self._find_index(C)] + r2 * self.values[self._find_index(D)] + r3 * self.values[self._find_index(A)]
        
        if self.verbose:
            print(f"Can't compute with given sample for point {y}")
        return float('NaN')
    
    def _find_index(self, point):
        """
        Find the index of the given point in the original points array.

        :param point: A tuple representing the point.
        :return: Index of the point in the points array.
        """
        return np.where((self.points == point).all(axis=1))[0][0]

    def plot(self, y):
        """
        Plot the points and the triangles used for interpolation.

        :param y: A tuple (y1, y2) representing the query point.
        """
        A, B, C, D = self.compute_closest_points(y)
        if A is None or B is None or C is None or D is None:
            print("Cannot find the necessary points.")
            return
        
        plt.figure(figsize=(8, 8))
        plt.scatter(self.points[:, 0], self.points[:, 1], color='blue', label='Points')
        plt.scatter([y[0]], [y[1]], color='red', label='y')
        plt.scatter([A[0]], [A[1]], color='green')
        plt.scatter([B[0]], [B[1]], color='purple')
        plt.scatter([C[0]], [C[1]], color='orange')
        plt.scatter([D[0]], [D[1]], color='pink')

        # Fill triangles with different colors for distinction
        plt.fill([A[0], B[0], C[0]], [A[1], B[1], C[1]], 'lightblue', alpha=0.5, edgecolor='black')
        plt.fill([C[0], D[0], A[0]], [C[1], D[1], A[1]], 'lightgreen', alpha=0.5, edgecolor='black')

        # Annotate the points A, B, C, D, and y
        plt.text(A[0], A[1], 'A', fontsize=12, ha='right', color='green')
        plt.text(B[0], B[1], 'B', fontsize=12, ha='right', color='purple')
        plt.text(C[0], C[1], 'C', fontsize=12, ha='right', color='orange')
        plt.text(D[0], D[1], 'D', fontsize=12, ha='right', color='pink')
        plt.text(y[0], y[1], 'y', fontsize=12, ha='right', color='red')

        plt.title('Barycentric Interpolation')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.grid(True)
        plt.show()
    
    def find_triangle_and_coordinates(self, y):
        """
        Find which triangle contains the point y and compute its barycentric coordinates.

        :param y: A tuple (y1, y2) representing the query point.
        :return: The name of the containing triangle and the barycentric coordinates, or an error message if no triangle is found.
        """
        A, B, C, D = self.compute_closest_points(y)
        if A is None or B is None or C is None or D is None:
            return "No triangle found", None, None
        
        if self.is_inside_triangle(y, A, B, C):
            coords = self.compute_barycentric_coordinates(y, A, B, C)
            return "ABC", coords
        
        if self.is_inside_triangle(y, C, D, A):
            coords = self.compute_barycentric_coordinates(y, C, D, A)
            return "CDA", coords
        
        return "No triangle found", None, None
