import random
import math
import matplotlib.pyplot as plt
import numpy as np
import json

verbose=True #debug control var
json_writing_indent = 5 #purely stylistic

storage_dir = "./storage/" #where to store files? this must exist already I did not create it here, so just make sure to create the folder first. absolute path fine, but
#im using relative path here. (careful if OS not windows, this might not work)

#assume alphabetical order when referencing arrays of sides or angles, that is:
#assume/ensure that all arrays of sides or angles look like [a,b,c], or [A,B,C]
#in this file currently capital letters are angles unless otherwise specified and lower case are sides.
#I realised this is backwards to what I beleive is typically taught in geometry but w/e we can fix it 
#later if you want.

class Triangle:
    """a simple class representing a triangle"""
    def __init__(self,A=1,B=1,C=1,a=60,b=60,c=60):
        """create a triangle with side lengths A,B,C and angles a,b,c
        everything defaults to a 1:1:1 isosolies triangle

        Args:
            A (float): side length A
            B (float): side length B
            C (float): side length C
            a (float): angle (in degrees) a
            b (float): angle (in degrees) b
            c (float): angle (in degrees) c
        """
        self.sides = [A,B,C]
        self.angles = [a,b,c]
    def sum_of_angles(self):
        """return a summation of all angles of this triangle

        Returns:
            float: sum of angles
        """
        return sum(self.angles)
        
    def validate_triangle(self):
        """validates if this triangle is valid or not, by checking all non-zero values, angles sum to 180 and the sum of any two sides are greater than the other side

        Returns:
            _type_: _description_
        """
        triangle_has_zero_length_sides = any([angle ==0 for angle in self.sides])
        triangle_has_zero_magnitude_angles = any([angle ==0 for angle in self.angles])
        triangle_has_no_zeros = not triangle_has_zero_length_sides and not triangle_has_zero_magnitude_angles
        sum_of_angles = self.sum_of_angles()
        angles_sum_to_180 = sum_of_angles >= 179.9 and sum_of_angles <= 180.1

        #this could be better, we could check every combination, but I dont see a reason to.
        sum_of_two_sides_greater_than_other = self.sides[0]+self.sides[1] > self.sides[2]
        
        #a triangle is valid if angles sum to 180, two sides are greater than other and the triangle has no zeros
        valid_triangle = angles_sum_to_180 and sum_of_two_sides_greater_than_other and triangle_has_no_zeros
        
        if(not valid_triangle and verbose):
            print(f"triangle_invalid:\ntriangle_has_no_zeros:{triangle_has_no_zeros}\nsum_of_angles:{sum_of_angles}\nangles_sum_to_180:{angles_sum_to_180}\nsum_of_two_sides_greater_than_other:{sum_of_two_sides_greater_than_other}")
        
        return valid_triangle
    
    def to_json(self):
        """convert this obj to a json string

        Returns:
            string: json string
        """
        json_string = json.dumps(self.__dict__, indent=json_writing_indent) #indent is just how many deep it will pretty print.
        return json_string
    
    def from_dict(self,dictionary):
        """read a dictionary and assign its values to this object

        Args:
            dictionary (dict): a dict equivalent to the json representation
            
        Returns: self
        """
        self.sides = dictionary["sides"]
        self.angles = dictionary["angles"]
        
        return self
    
    def from_json(self, json_string):
        """read a json string and assign its values to this object

        Args:
            json_string (string): the string containing the json data
            
        Returns: self
        """
        data = json.loads(json_string)
        self.from_dict(data)
        return self
    
    def draw_triangle(self):
        #yoinked from here https://stackoverflow.com/questions/70050740/trouble-plotting-a-right-triangle-at-an-angle-in-matplotlib to help visualize
        C_radians = np.radians(self.angles[2])

        Cx = self.sides[1] * np.cos(C_radians)
        Cy = self.sides[1] * np.sin(C_radians)

        vertices_x = [0, self.sides[0], Cx, 0]
        vertices_y = [0, 0, Cy, 0]

        plt.plot(vertices_x, vertices_y, marker='o')
        plt.xlabel("x-axis")
        plt.ylabel("y-axis")
        plt.title("Triangle with sides a, b and angle C")
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
        
    def __str__(self):
        """yield the json rep when converted to string"""
        return self.to_json()
    
class Triangle_Array:
    """a wrapper class for multiple triangles"""
    def __init__(self, array=[]):
        """

        Args:
            array (Triangle[]): array of triangles
        """
        self.array = array
        
    def to_file(self,file_name="triangle_array"):
        """write an existing array of triangles to a file

        Args:
            triangle_array (Triangle[]): the array of triangles to write
            file_name (str, optional): file name to write to, .json can be included or left absent. Defaults to "triangle_array".
        """
        if not ".json" in file_name: file_name += ".json"  #add extention if not present
        triangle_dict = {"triangles": [triangle.__dict__ for triangle in self.array]}
        with open(storage_dir+file_name, 'w') as file:
            file.write(json.dumps(triangle_dict,indent=json_writing_indent))
        return self
        
    def from_file(self,file_name="triangle_array"):
        """load a triangle array from a file

        Args:
            file_name (str, optional): name of file to load. Defaults to "triangle_array".
        """
        if not ".json" in file_name: file_name += ".json"  #add extension if not present
        triangle_dict= {}
        with open(storage_dir+file_name, 'r') as file:
            data = file.read()
            triangle_dict=json.loads(data)
        if(verbose): print(f"loading triangle array, dict rep: {triangle_dict}")
        triangle_dict_array = triangle_dict["triangles"] #the array of triangle dictionaries
        triangle_array = [Triangle().from_dict(dict) for dict in triangle_dict_array]
        if(verbose): print(f"loaded array: {triangle_array}")
        self.array = triangle_array
        return self
        
class TriangleGenerator:
    """a wrapper class for triangle for generating valid triangles for our data set and storing/reading them to/from disk
    """
    def __init__(self,min_side_length =1,max_side_length=10,force_right_triangles_only=False):
        self.min_side_length = min_side_length
        self.max_side_length = max_side_length
        self.force_right_triangles_only = force_right_triangles_only
        
    def generate_array_of_N_triangles(self,n):
        """return an array of N triangles

        Args:
            n (int): how many triangles

        Returns:
            Triangle[]: array of n triangles
        """
        return Triangle_Array([self.generate_random_triangle_via_angles() for x in range(0,n)])
    
    def generate_random_triangle_via_angles(self):
        """create a new triangle by randomly setting the angles then solving for the side lengths"""
        angle_a = random.randrange(1, 90) if not self.force_right_triangles_only else 90 #force 90 degrees if enabled, otherwise roll 
        remaining_possible_angle_magnitude = 180 - angle_a
        angle_b = random.randrange(1,remaining_possible_angle_magnitude-1)
        angle_c = 180 - angle_a - angle_b
        
        #side from point a to point b
        side_A_B = random.randrange(self.min_side_length, self.max_side_length)
        
        side_B_C = math.sin(math.radians(angle_a))*(side_A_B/math.sin(math.radians(angle_c)))
        side_A_C = math.sin(math.radians(angle_b))*(side_A_B/math.sin(math.radians(angle_c)))
        
        
        angles = [angle_a, angle_b, angle_c]
        sides = [side_A_B,side_B_C,side_A_C]
        
        triangle = Triangle(*sides, *angles)
        if(verbose):
            json_string = triangle.to_json()
            print(f'-----new triangle------\n -(generate_random_triangle_via_angles)-\n')
            print(f"angles:{angles}\nsides:{sides}")
            print(f"valid triangle? {triangle.validate_triangle()}")
            print(f"json representation: {json_string}")
            print("---triangle_gen END---\n")
            triangle.from_json(json_string)
    
        return triangle
    
    def load_array_of_triangle_from_file(self,file_name="triangle_array"):
        """
        load an array of triangles from a file returning the Triangle_Array"""
        if not ".json" in file_name: file_name += ".json"  #add extension if not present
        return Triangle_Array().from_file(file_name)
    
    def generate_array_and_write_to_file(self,N,file_name="triangle_array"):
        """generate an array of triangles of N length and write to file as a json

        Args:
            N (int): how many triangles?
            file_name (str, optional): what should the file be named, you may include or forgo .json extension. Defaults to "triangle_array".
        """
        return self.generate_array_of_N_triangles(N).to_file(file_name)
    
def main():
    """a driver function to test this portion of code"""
    tri_gen = TriangleGenerator()
    right_angle_tri_gen = TriangleGenerator(force_right_triangles_only=True)
    #generate a single triangle and draw it
    tri_gen.generate_random_triangle_via_angles().draw_triangle()
    #generate a single right triangle and draw it
    right_angle_tri_gen.generate_random_triangle_via_angles().draw_triangle()
    
    #seems like the script to draw triangles is not perfect... :( does not seem to work right
    
    
    #generate an array of triangles and write it to a file
    triangles_before_write = tri_gen.generate_array_and_write_to_file(5, "triangle_gen_demo.json")
    
    #draw first triangle
    triangles_before_write.array[0].draw_triangle() 
    
    #read previously generated array from file
    triangles = tri_gen.load_array_of_triangle_from_file("triangle_gen_demo")
    
    #draw the first triangle again
    triangles.array[0].draw_triangle()
    
    print(f"side lengths of written and read array are same: {triangles.array[0].sides == triangles_before_write.array[0].sides}")

if __name__ == "__main__":
    main()
