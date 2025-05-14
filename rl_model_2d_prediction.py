import gymnasium as gym
from gymnasium import spaces
import rl_model_base
import numpy as np
import triangle_gen
import sklearn.metrics
import random
import math

random.seed(42)

verbose_in_training = False #should verbose also be on durring training (NO)
verbose=False #print way too much stuff to terminal

def x_plus_10(x):
    return x +10

class RL_2D_Predict(rl_model_base.RL_ENV):
    """this extends the base one and is used for predicting one value off another value
    """
    
    iso_lambda = lambda: (lambda r: triangle_gen.PartialTriangle([60, 60, 60, r, r, r], known_indices=[0,1,2,3]))(random.random() * 10)
    right_angle_lambda = lambda: (lambda x: (lambda r: triangle_gen.PartialTriangle([90, x, np.nan, np.nan, np.nan, r], known_indices=[0,1,5]))(random.random() * 10))(random.random() * 89)
    
    def __init__(self, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.9, function_to_predict=x_plus_10, smallest_value_allowed=0.0001, base_increment_size =1):
        super().__init__()
        self.smallest_value_allowed = smallest_value_allowed
        self.base_increment_size = base_increment_size
        self.increment_size = self.base_increment_size
        
        self.function_to_predict = function_to_predict
        
        #2 actions, increment or decrement, and adjust scaler
        self.action_space = spaces.MultiDiscrete([4,2]) 
        #the last two actions are to increese or decreese the increment size by 10%
        
        self.reward_for_solving = 1000 * self.reward_scaler #a large reward provided if a step creates a valid triangle
        self.reward_for_step = -0.6 #a negative reward to decentivize taking a lot of steps, this applies pretty much always
        self.reward_for_good_step = 2 # a small psoitive reward if we got closer to the correct answer
        
        #just the two 
        self.observation_space = spaces.Box(
            low=np.array([0,0]),
            high=np.array([np.inf,np.inf]),
            dtype=np.float32
        )
        
        #setup initial state
        self.reset()

    
    def get_error(self,x,y_guess):
        """return percent error of our geuss

        Args:
            x (_type_): _description_
            y_guess (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_real = self.function_to_predict(x)
        percent_error = np.abs((y_real - y_guess))
        return percent_error
        
        
    def reset(self, predefined_value=None, seed=None, options=None):
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
        
        self.state = predefined_value if predefined_value != None else np.array([random.random()*10,np.nan])
        #print(f"state:{self.state}")
        return self.state, {}
    
    def perform_step(self,action):
        """this function handles actually executing a step and should be overwridden with the specific functionality of the action set for the agent

        Args:
            action (_type_): _description_
        """
        
        index, sub_action = action
        
        match index:
            case 1: #increment/decrement y
                sign = 1 if sub_action == 1 else -1

                if np.isnan(self.state[1]):
                    self.state[1] = self.smallest_value_allowed
                
                self.state[1] += self.increment_size * sign
            case 2: #scale increment by 10% up or down
                sign = 1 if sub_action == 1 else -1
                if(sign == 1):
                    self.increment_size*=1.1
                else:
                    self.increment_size*=.9
            case 3: #scale increment by 200% up or down
                sign = 1 if sub_action == 1 else -1
                if(sign == 1):
                    self.increment_size*=2
                else:
                    self.increment_size*=.5
            case 4: #raise increment to power of 2 or .5
                sign = 1 if sub_action == 1 else -1
                if(sign == 1):
                    self.increment_size**2
                else:
                    self.increment_size**.5
          
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
        
        if(self.get_error(self.state[0], self.state[1]) < self.get_error(prev_state[0], prev_state[1])):
            reward += self.reward_for_good_step
            
        if(self.get_error(self.state[0], self.state[1]) < self.threshold_for_solving):
            reward += self.reward_for_solving
            done = True
            if(verbose): print("!!reward for solving!!")
        
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
    env = RL_2D_Predict()
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
        