import random
import math
import matplotlib.pyplot as plt
import numpy as np
import json
from enum import Enum

random.seed(42)

verbose=False #debug control var
json_writing_indent = 5 #purely stylistic

storage_dir = "./storage/" #where to store files? this must exist already I did not create it here, so just make sure to create the folder first. absolute path fine, but
#im using relative path here. (careful if OS not windows, this might not work)

#assume alphabetical order when referencing arrays of sides or angles, that is:
#assume/ensure that all arrays of sides or angles look like [a,b,c], or [A,B,C]
#in this file currently capital letters are angles unless otherwise specified and lower case are sides.
#I realised this is backwards to what I beleive is typically taught in geometry but w/e we can fix it 
#later if you want.


class TriangleType(Enum):
    """a simple enum to classify what type of triangle we have"""
    AAA = 0
    AAS = 1
    ASA = 2
    SAS = 3
    SSA = 4
    SSS = 5
    UNSOLVABLE = 6

class Triangle:
    """a simple class representing a triangle"""
    def __init__(self,a=60,b=60,c=60,A=1,B=1,C=1,type=False):
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
        self.type= type
        
    
    def sum_of_angles(self):
        """return a summation of all angles of this triangle

        Returns:
            float: sum of angles
        """
        return sum(self.angles)

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
        plt.title("Triangle")
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
    
    def generate_random_triangle_via_angles(self,known_indices=[]):
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
        
        triangle = PartialTriangle(*sides, *angles, known_indices=known_indices)
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

    
class PartialTriangle(Triangle):
    """a class for triangles with incomplete data

    Args:
        Triangle (_type_): _description_
    """
    
    #classifier for intended way to solve:
    #AAA -- not solution possible
    #AAS -- 180-a-b then law of sines (to find other sides)
    #ASA -- 180-a-b then law of sines (to find other 2 sides)
    #SAS -- law of cosines then law of sines then 180-a-b for last angle
    #SSA -- law of sines (to calculate 1 angle) then 180-a-b then law of sines (to find last side)
    #SSS -- arc cosign or arc sin
    
    # let 180-a-b = 0
    # let law of sines = 1
    # let law of cosines = 2
    # let arc sin = 3
    # let arc cosin = 4
    
    # then our intended solution vector for a SSA triangle would be [1,0,1] implying that we would need sin, sum to 180 then sin.
    
    #okay so for us to classify what the intended solution is we must first classify our data into AAA, AAS, ASA, SAS or SSA, or SSS, Or None, none being unsolvable honestly that might be
    #a better thing to tell our RL model then the intended solution vector, just telling it what type of triangle it is.
    
    def __init__(self, A=1, B=1, C=1, a=60, b=60, c=60, type=None, known_indices = []):
        
        if(isinstance(A, list)):
            B = A[1]
            C = A[2]
            a = A[3]
            b = A[4]
            c = A[5]
            A = A[0]
            
        super().__init__(A, B, C, a, b, c)
        #now our triangle is made, the actual and correct values are stored, that is the ground truth exists, now we must choose what values are missing
        
        self.incomplete_sides = []
        self.incomplete_angles = []
        
        
        
        if(isinstance(A, Triangle)):
            self = A
            return
             
        
        
        maxSidesMissing = 2
        maxAnglesMissing = 2
        maxValuesMissing = 3
        
        sidesMissing = 0
        anglesMissing = 0
        
        if(len(known_indices) == 0):
            for angle in self.angles:
                _angle = angle
                if(random.random() > .5 and anglesMissing < maxAnglesMissing and sidesMissing+anglesMissing<maxValuesMissing):
                    _angle = 0
                    anglesMissing+=1
                self.incomplete_angles.append(_angle)
                
            for side in self.sides:
                _side = side
                if(random.random() > .5 and sidesMissing < maxSidesMissing and sidesMissing+anglesMissing<maxValuesMissing):
                    _side = 0
                    sidesMissing+=1
                self.incomplete_sides.append(_side)
        else:
            for index in range(0,6):
                if(index>2):
                    #if greater than 2 then this is a side
                    side_index = index-4
                    # print(f"side index: {side_index}")
                    # print(f"side array: {self.sides}")
                    if(index in known_indices):
                        self.incomplete_sides.append(self.sides[side_index])
                    else:
                        self.incomplete_sides.append(np.nan)
                else:
                    #if less than or equal to 2 this is an angle
                    if(index in known_indices):
                        self.incomplete_angles.append(self.angles[index])
                    else:
                        self.incomplete_angles.append(np.nan)
                    
                
            
        self.triangle_type = self.get_type()
        #print(self.triangle_type)
    
    def validate_triangle(self,state):
        """validates if this triangle is valid or not, by checking all non-zero values, angles sum to 180 and the sum of any two sides are greater than the other side

        Returns:
            _type_: _description_
        """
        return np.isclose(self.get_error(state), 0, atol=0.1)
        
    def get_type(self):
        angle_exists_array = [1 if side>0 else 0 for side in self.incomplete_angles]
        side_exists_array = [1 if side>0 else 0 for side in self.incomplete_sides]
        exists_array = angle_exists_array + side_exists_array
        #what im thinking is convert the inclusing array to a binary string then to an int to map each possible inclusion state to a int number
        return int("".join(str(bit) for bit in exists_array),2) #base 2
    
    def get_error(self,state):
        #the idea here is to use law of sines to find how much error we have compared to a correct answer
        A,B,C = state[:3] #angles
        a,b,c = state[3:6] #sides
        
        #print(f"get err: {A} {B} {C} {a} {b} {c}")
        
        # Calculate the angle error (how close the angles are to 180 degrees)
        angle_error = np.abs(180 - (A + B + C))
        
        # Calculate the side error
        side_error = np.abs(sum(np.abs(self.sides[i]-np.abs(state[3:6][i])) for i in range(0,3)))
        #print(f"sides:{self.sides}, geussed sides: {state[3:6]}")
        #print(side_error)
        
        # Combine angle error and side error into a single metric
        total_error = angle_error + side_error
        #print(f"error:{total_error}")
        return total_error
    
    def get_state_rep(self):
        return self.incomplete_angles + self.incomplete_sides + [self.triangle_type]
            
        
            
            
    
    def classify(self):
        """
        this method will classify the triangle that we have into AAA, AAS, SAS, SSA or SSS
        based on what angle/side array elements are missing, we denote a missing value by a 0
        """
    
        #first check if its solvable at all, that is 
        
        #get arrays of where each element is 1 or 0 badsed on if we have that angle or not
        angle_exists_array = [1 if side>0 else 0 for side in self.incomplete_angles]
        side_exists_array = [1 if side>0 else 0 for side in self.incomplete_sides]
        
        #find the sums
        side_count = sum(side_exists_array)
        angle_count = sum(angle_exists_array)
        
        
        
        has_two_angles_and_one_side = side_count >= 2 and angle_count >=1
        has_two_sides_and_one_angle = side_count >=1 and angle_count >=2
        has_three_sides = side_count >= 3
        
        #if none of the previous conditions are met there is no solution
        is_unsolvable = not has_two_angles_and_one_side and not has_two_sides_and_one_angle and not has_three_sides
        
        #angle angle angle is also unsolvable but we will clasify it anyways
        is_AAA = angle_count >=3 and side_count == 0
        
        #side angle side would be two sides that are not oposite to an angle and one angle that is not oposite to a side
        is_SAS = sum([angle_exists_array[i] ^ side_exists_array[i] for i in range (0,len(self.incomplete_sides))]) == 3 and side_count == 2 and angle_count == 1
        
        #angle side angle would mean a side in between two angles, for instance angles=[1,1,0], sides=[0,0,1] or a[0,1,1] s[1,0,0] or a[1,0,1] s[0,1,0] seems like we could just xor the two arrays
        #I am going to leave ASA broad just checking if it is viable to solve the triangle like an ASA triangle rather than validating it is solely an ASA triangle
        #that is if there is extra info but it can still be solved like an ASA then we will clasify it as an ASA triangle, the rest will be exclusive thus this functions as
        #the catch all for triangles with a lot of extra info
        #we will also check if its not a SAS triangle as these two are the same check other than the count of sides and angles
        is_ASA = sum([angle_exists_array[i] ^ side_exists_array[i] for i in range (0,len(self.incomplete_sides))]) == 3 and not is_SAS
        
        #angle angle side would be two angles followed by a side, that is, we have the side opposite to atleast one of our angles
        is_AAS = sum([angle_exists_array[i] & side_exists_array[i] for i in range (0,len(self.incomplete_sides))]) == 1 and angle_count == 2 and side_count==1
        
        #side side angle would be two sides followed by an angle, that is, we have the one side opposite to one angle and one side not opposite
        is_SSA = sum([angle_exists_array[i] & side_exists_array[i] for i in range (0,len(self.incomplete_sides))]) == 1 and angle_count == 1 and side_count==2
        
        is_SSS = side_count == 3 and angle_count == 0
        
        if(is_SSS):
            return TriangleType.SSS
        elif(is_SSA):
            return TriangleType.SSA
        elif(is_AAS):
            return TriangleType.AAS
        elif(is_ASA):
            return TriangleType.ASA
        elif(is_AAA):
            return TriangleType.AAA
        elif(is_SAS):
            return TriangleType.SAS
        elif(is_unsolvable):
            return TriangleType.UNSOLVABLE
        
        
        
        
        
        
    
    
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
    triangles_before_write = tri_gen.generate_array_and_write_to_file(100, "triangle_gen_demo.json")
    
    #draw first triangle
    triangles_before_write.array[0].draw_triangle() 
    
    #read previously generated array from file
    triangles = tri_gen.load_array_of_triangle_from_file("triangle_gen_demo")
    
    #draw the first triangle again
    triangles.array[0].draw_triangle()
    
    print(f"side lengths of written and read array are same: {triangles.array[0].sides == triangles_before_write.array[0].sides}")

if __name__ == "__main__":
    main()
    tri = PartialTriangle()
    print(f"triangle sides: {tri.incomplete_sides}")
    print(f"triangle angles: {tri.incomplete_angles}")
    print(tri.triangle_type)
    #tri.draw_triangle()


