import gymnasium as gym
from gymnasium import spaces
import rl_model_base
import numpy as np
import triangle_gen
import sklearn.metrics
import random
import math

verbose_in_training = False #should verbose also be on durring training (NO)
verbose=False #print way too much stuff to terminal

class RL_Triangle_Predict_Model_2(rl_model_base.RL_ENV):
    """this extends the base one and is used for predicting one value off another value
    """
    
    iso_lambda = lambda: (lambda r: triangle_gen.PartialTriangle([60, 60, 60, r, r, r], known_indices=[0,1,2,3]))(random.random() * 10)
    right_angle_lambda = lambda: (lambda x: (lambda r: triangle_gen.PartialTriangle([90, x, np.nan, np.nan, np.nan, r], known_indices=[0,1,5]))(random.random() * 10))(random.random() * 89)
    
    def __init__(self, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.9):
        super().__init__()
        
        
        #the last two actions are to increese or decreese the increment size by 10%
        
        self.reward_for_solving = 1000 * self.reward_scaler #a large reward provided if a step creates a valid triangle
        self.reward_for_step = -0.6 #a negative reward to decentivize taking a lot of steps, this applies pretty much always
        self.reward_for_good_step = 2 # a small psoitive reward if we got closer to the correct answer
        
        self.largest_value_allowed_for_sides = np.inf
        self.data_type = np.float32 #the datatype of our observation space, i figure 32 bit float is fine for now
        
        #I defined the largest value for angles to be 180- 2 of the smallest value allowed as that would be required for it to be a valid triangle. idk if that is cheating
        self.largest_value_allowed_for_angles = 180-(self.smallest_value_allowed*2)
        self.min_triangle_type_value = 0 #the minimum number for tryangle type feature
        self.max_triangle_type_value = 2**6 #each side or angle can be present or not present thus 2 choices raised to the number of slots gets us the 64 possible types


        #actions
        self.action_space = spaces.MultiDiscrete([9,6,9]) 
        self.action_space_shape = [9,6,9]
        
        #state definition:
        #[angle a, angle b, angle c, side a, side b, side c, triangle_type, scratch,scratch]
        self.observation_space = spaces.Box(
            low=np.array([1 for n in range(0,6)] + [self.min_triangle_type_value] +[-np.inf,-np.inf]),
            high=np.array(
                [
                self.largest_value_allowed_for_angles,self.largest_value_allowed_for_angles,self.largest_value_allowed_for_angles, #angles
                self.largest_value_allowed_for_sides,self.largest_value_allowed_for_sides,self.largest_value_allowed_for_sides, #sides
                self.max_triangle_type_value, #triangle type,
                np.inf,np.inf
                ]
            ),
            dtype=self.data_type
        )
        
        #setup initial state
        self.reset()

    
    def get_error(self,state):
        """return percent error of our geuss

        Args:
            x (_type_): _description_
            y_guess (_type_): _description_

        Returns:
            _type_: _description_
        """

        return self.real_triangle.get_error(state[0:7])
        
        
    def reset(self, predefined_triangle=None, seed=None, options=None):
        """this is the reset function, it is the way that the RL agent resets after an episode of training to a base state.
            I do not know what options are but the base env class wants them for extending its reset method so I have them there.
        Args:
            predefined_triangle (float32[6]): an array of 6 float values in the pattern [angle,a,a,side,s,s] to specify a specific triangle to solve. Defaults to None.
            seed (_type_, optional): _description_. Defaults to None.
            options (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.state = super().reset(seed=seed, options=options)
        
        if(predefined_triangle):
            self.state = predefined_triangle.get_state_rep() + [0,0]
            self.real_triangle = predefined_triangle
        else:
            #if none is provided we will create one
            triangle = triangle_gen.TriangleGenerator().generate_random_triangle_via_angles()
            self.real_triangle = triangle
            triangle_state = triangle.get_state_rep() + [0,0]
            self.state = np.array(triangle_state, dtype=self.data_type)
            
            #choose up too 3 indecies to be missing
            missing_indices = np.random.choice(6, size=np.random.randint(1, 3), replace=False)
            for index in missing_indices:
                #set them to not a number
                self.state[index] = np.nan
                
        #our inital state is now generated or directly specified, now find our known values and add them to the list
        for n in range(0,len(self.state)):
            if(not np.isnan(self.state[n])):
                self.known_indices.append(n)
        return self.state, {}
    
    def perform_step(self,action):
        """this function handles actually executing a step and should be overwridden with the specific functionality of the action set for the agent

        Args:
            action (_type_): _description_
        """
        
        rhs_index, sub_action, lhs_index = action
        
        prev_state = self.state.copy()
        
        #if the agent makes a invalid move, that is if it edits a known value, we provide a negative reward and end this step
        if rhs_index in self.known_indices or rhs_index==6:
            return self.state, self.reward_for_illegal_action, False 
        
        match sub_action:
            case 1: #addition
                self.state[rhs_index] += self.state[lhs_index] 
            case 2: #subtraction
                self.state[rhs_index] -= self.state[lhs_index] 
            case 3: #multiplication
                self.state[rhs_index] *= self.state[lhs_index]
            case 4: #division
                self.state[rhs_index] /= self.state[lhs_index]
            case 5: #exponent
                self.state[rhs_index] **= self.state[lhs_index]
            case 6: #xor
                self.state[rhs_index] ^= self.state[lhs_index]
                
        #no negatives in sides or angles
        if(rhs_index < 5 and self.state[rhs_index]) <0:
            self.state = prev_state
            return self.state, self.reward_for_illegal_action, False 
          
        return self.state, None, False
    
    def get_reward(self, action, prev_state):
        """this function handles any reward that should be given after the action is performed, it is added into the total reward
        override this with the reward policies for the model

        Args:
            action (_type_): the action that was performed
            prev_state (_type_): the previous state before the action was preformed
        Returns:
            tuple: state, reward, is_done
        """
        reward = 0
        done = False
        
        #--reward for good step--
        if(self.get_error(self.state) < self.get_error(prev_state)):
            reward+=self.reward_for_good_step 
            
        #--reward for solving--
        if(self.real_triangle.validate_triangle(self.state)):
            reward+=self.reward_for_solving
            done = True
        
        return self.state, reward, done
        
    def __str__(self):
        """just so we can easily print it to terminal"""
        return f"state:{self.state}\nknown indexs:{self.known_indices}\n"
    
    
def main():
    #now lets try to solve a triangle, a equilateral triangle will have all sides of the same length, as such it should find [60,60,60,5,5,5]
    solved_states = []
    errorArray = []
    train_count = 100
    test_count = 100
    env = RL_Triangle_Predict_Model_2()
    env.train(train_count)
    #find 100 sample problems to solve
    for n in range(test_count):
        state = env.solve(None)
        solved_states.append(state)
        errorArray.append(env.get_error(state[0],state[1]))
    print(errorArray)
    print(solved_states)
    
    #calculate the error of each of these, since they are isosolises in this case its just going to be how far off the sides are from the 1 provided side

    

    mean_squared_error = (sum([error**2 for error in errorArray])/len(solved_states))
    
    print(f"MSE:{mean_squared_error}")
    

if __name__ == "__main__":
    main()  
        