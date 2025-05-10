import gymnasium as gym
from gymnasium import spaces
import numpy as np
import triangle_gen
import sklearn.metrics
import random

verbose_in_training = False #should verbose also be on durring training (NO)
verbose=False #print way too much stuff to terminal

class TriangleRL_Environment(gym.Env):
    """this is the environment of our RL model, this can be thought of as the game that our agent is trying to win. So the state is like what exists in the game,
    the various different methods and such are the actions it can take. An action is taken when the step() function is called with an action to take
    
    We must extend gym's env class, I have not looked enough into all the things it does yet to know fully why that is mandatory but the docs
    heavily imply it is.
    """
    
    iso_lambda = lambda: (lambda r: triangle_gen.PartialTriangle([60, 60, 60, r, r, r], known_indices=[0,1,2,3]))(random.random() * 10)
    right_angle_lambda = lambda: (lambda x: (lambda r: triangle_gen.PartialTriangle([90, x, np.nan, np.nan, np.nan, r], known_indices=[0,1,5]))(random.random() * 10))(random.random() * 89)
    
    def __init__(self, exploration_rate=0.1, learning_rate=0.1, discount_factor=0.9):
        super(TriangleRL_Environment, self).__init__()
        
        #q-learning vars
        self.exploration_rate = exploration_rate
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}
        
        
        #config variables for reward policies
        self.reward_scaler = 1 #applied globally to all rewards
        self.reward_for_illegal_action = -10 * self.reward_scaler # the negative reward to decentivize illigal actions (modifying known values)
        self.reward_for_no_NAN = .01 * self.reward_scaler #this reward is given for every step that has all NAN values gone. this might not be good to have, mark as 0 to disable
        self.reward_for_valid_triangle = 100 * self.reward_scaler #a large reward provided if a step creates a valid triangle
        self.reward_for_step = -0.5 #a negative reward to decentivize taking a lot of steps, this applies pretty much always
        self.reward_for_good_step = .5 # a small psoitive reward if we got closer to the correct answer
         
        
        #a variable that will help bound our observation state later
        self.smallest_value_allowed = .01
        #the rounding that we do should be to the same decimal amount as the smallest value the RL agent is allowed to use
        self.number_of_decimals_for_discrete_space = len(str(self.smallest_value_allowed).split(".")[1]) 
        
        self.largest_value_allowed_for_sides = np.inf
        self.data_type = np.float32 #the datatype of our observation space, i figure 32 bit float is fine for now
        
        #I defined the largest value for angles to be 180- 2 of the smallest value allowed as that would be required for it to be a valid triangle. idk if that is cheating
        self.largest_value_allowed_for_angles = 180-(self.smallest_value_allowed*2)
        
        self.min_triangle_type_value = 0 #the minimum number for tryangle type feature
        self.max_triangle_type_value = 2**6 #each side or angle can be present or not present thus 2 choices raised to the number of slots gets us the 64 possible types
        
        #the amount that the agent will be able to add or subtract from a side/angle
        self.base_increment_size = 1
        self.increment_size = self.base_increment_size
        
        
        #note I decided to change our methodology, we will leave unknowns as NAN to start with, this will be easier to deal with than saying 0.
        #our state itself is constrained to the 3 sides and 3 angles for now. As such it is just a 6 long array
        self.state = np.full(6,np.nan)
        
        #The action space for now I have defined very simply, for each element in the state the agent may add or subtract a small value
        #This means we have 12 total actions for this simple prototype
        #we use multidiscrete as our action space type, that is there are 6 discrete categories that each have 2 discrte actions (+/-)
        self.action_space = spaces.MultiDiscrete([7,2]) 
        #the last two actions are to increese or decreese the increment size by 10%
        
        #now we must define the observation space, that is: the set of all possible states that the agent could observe.
        #in our case we have 6 continous values with a few bounds.
        #angles obviously must be bounded between positive number to ~180 and sides must be bounded between small positive number and infinity
        #the first 3 values will be angles, second 3 will be sides
        self.observation_space = spaces.Box(
            low=np.array([1 for n in range(0,6)] + [self.min_triangle_type_value]),
            high=np.array(
                [
                self.largest_value_allowed_for_angles,self.largest_value_allowed_for_angles,self.largest_value_allowed_for_angles, #angles
                self.largest_value_allowed_for_sides,self.largest_value_allowed_for_sides,self.largest_value_allowed_for_sides, #sides
                self.max_triangle_type_value #triangle type
                ]
            ),
            dtype=self.data_type
        )
        
        
        #step controls
        self.max_steps = 1000
        self.steps = 0
        
        #this will be the list of indices, that is the sides and angles that were given, in other words, the already known sides and angles, thus the ones
        #the agent should not be allowed to edit. Of course we will not directly disallow it from editing them, rather just negatively reward it if it does.
        self.known_indices = []
        
        #setup initial state
        self.reset()
        
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
        super().reset(seed=seed, options=options)
        
        self.increment_size = self.base_increment_size
        
        #we reset our list of known indicies to empty
        self.known_indices = []
        #reset #steps to 0
        self.steps = 0
        
        #if user provided a triangle use it
        if(predefined_triangle):
            self.state = predefined_triangle.get_state_rep()
            self.real_triangle = predefined_triangle
        else:
            #if none is provided we will create one
            triangle = triangle_gen.TriangleGenerator().generate_random_triangle_via_angles()
            self.real_triangle = triangle
            triangle_state = triangle.get_state_rep()
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
        #print(f"known indecies:{self.known_indices}")
        #print(self.state)
        
        return self.state, {}
        
    def __str__(self):
        """just so we can easily print it to terminal"""
        return f"state:{self.state}\nknown indexs:{self.known_indices}\n"
    
    def step(self, action):
        """this is the method that allows the rl agent to edit its state and interact with the "world" 
        it can take any of the actions defined in the action state specified about

        Args:
            action (tuple(6,2)): the action it will take, the first number in the tuple is the index of its state to alter, the second is to add or subtract from it.
            
        Returns:
            tuple: new state, reward, is_done
        """
        #print(self.state)
        #incremet step cntr, seems fine to incremeent first here.
        self.steps+=1
        
        #if we hit max steps give a large negative reward, return our state and mark done to true
        if(self.steps > self.max_steps):
            return self.state, -10, True
        
        #if we are not at max steps then cont
        
        #unpack our action, index is the index of state to modify, sub action in this is currently just 0 or 1, 0 for subtract, 1 for add
        index, sub_action = action
        
        #Make a copy of our state for later to see if we improved or nah
        prev_state = self.state.copy() #might need __deepcopy__ if copy dosnt work right
            
        if(index < 6):
            
            #figure out if we are adding or subtracting
            sign = 1 if sub_action == 1 else -1
            
            
            
            #if the agent makes a invalid move, that is if it edits a known value, we provide a negative reward and end this step
            if index in self.known_indices:
                return self.state, self.reward_for_illegal_action, False 
            
            #if the agent sets a side or angle to a negative value this is also illegal
            if(self.state[index] + self.increment_size * sign) <0:
                return self.state, self.reward_for_illegal_action, False 
            
            #if this index is yet to be defined we will define it as the smallest allowed value to allow it to be modified
            if np.isnan(self.state[index]):
                #print("valid act")
                self.state[index] = self.smallest_value_allowed
            
            #perform action
            self.state[index] += self.increment_size * sign
            
            
            #print(f"step {self.step} state: {self.state}")
        else:
            #if greater than 6 than this is the other action of increesing increment count
            sign = 1 if sub_action == 1 else -1
            if(sign == 1):
                self.increment_size*=1.1
            else:
                self.increment_size*=.9
        
        reward =self.reward_for_step
        done = False
        
        if(not np.isnan(self.state).any()):
            #we will provide a small reward for eliminating the NAN values, this might not be a good thing to do
            #so we can disable it if its not good
            reward+=self.reward_for_no_NAN
        if(self.real_triangle.validate_triangle(self.state)):
            #if the triangle is now valid, we will provide a large reward and mark done to true
            reward+=self.reward_for_valid_triangle
            done = True
        
            
        #and finally we want to check if this state is better than the previous state, if it is (that is if the error is smaller as calculated in triangle_gen) lets give a small reward
        if(self.real_triangle.get_error(self.state) < self.real_triangle.get_error(prev_state)):
            reward+=self.reward_for_good_step 
            
        #return our stuff
        return self.state,reward,done
            

        
        #incremet step cntr
        #self.steps+=1
        
    def _init_q_table_row_if_not(self,tuple):
        """a simple helper function taht initalizaes a qtable row all to 0 if the row is undefined
        or I should say updates the dict entry to contain an array of 12 zeros

        Args:
            tuple (_type_): _description_
        """
        
        #we need to init values if we havent seen them before. There are twelve actions possible so 12 0's
        if tuple not in self.q_table:
            #TODO: this should nto be hardcoed 12
            self.q_table[tuple] = np.zeros(14)
        
    def find_next_action_pls(self,state):
        """choose our state based on the q table values

        Args:
            state (_type_): _description_
        """
        
        #convert our current state into something that we can use as a dictionary key, we are rounding to 2 decimal places here as to techinicalyl discretize the state space
        state_tuple = tuple(np.round(state, self.number_of_decimals_for_discrete_space)) 
        
        #we need to init values if we havent seen them before. There are twelve actions possible so 12 0's
        self._init_q_table_row_if_not(state_tuple)
            
        if(np.random.rand() < self.exploration_rate):
            #if we choose to explore
            action = self.action_space.sample() #grab a random action from the bag of actions
        else:
            #if we choose not to explore that means we need to behave inline with our q_table, that is we must choose the maximum value from the current values in the state that we are in
            #print(f"max:{max(self.q_table[state_tuple])}")
            
            action = np.argmax(self.q_table[state_tuple])
            
            #but since we expressed the qtable as 12 values we need to map the value 0-11 back into a tuple of len (6,2)
            act_index = action // 2 #int division to split in half
            direction = action % 2 #modulus to find +/-, 

            #reassemble
            action = (act_index, direction)
            
        #and now return our cool new action
        return action
    
    def q_learning_handler(self, state, action, reward, next_state):
        """the function that updates and handles the qlearning table 

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            next_state (_type_): _description_
        """
        #once more we will convert these to tuples to be sure cus it didnt work without this and fuck idk ill figure it out ltr
        state_tuple = tuple(np.round(state, 2))
        next_state_tuple = tuple(np.round(next_state, 2))
        
        #we need to init values if we havent seen them before. There are twelve actions possible so 12 0's
        self._init_q_table_row_if_not(state_tuple)
        #and for the next state to be safe
        self._init_q_table_row_if_not(next_state_tuple)
        
        
        #now we need to convert our action into a single number 0-11 reverse of what we did above. this is quite simple, its just going to be the index times 2 and then add back in the direction
        target_index = action[0] 
        direction = action[1]
        action_after_mapping = target_index * 2 + direction
        
        
        if(action_after_mapping < 0 or action_after_mapping > 13):
            print("wtf that should not happen. go look at line 259 you fucked up the mapping function some how idk like this should not be thrown I dont know how this would be thrown but ill put a msg here in case it does and im madge later")
            
            
        #okay, so now for the q learning part itself
        #we define a q value as (1-a)*c + a * (r + (L * maxQ))
        #where
        #a is self.learning rate
        #c is the current q value for this choice -> self.q_table[state_tuple][action_after_mapping]
        #r is the reward from the action
        #L is the discount factor (idk how to type lambda) -> self.discount value
        #maxQ is the estimate of the optimal future value
        c = self.q_table[state_tuple][action_after_mapping]
        a= self.learning_rate
        L = self.discount_factor
        r= reward
        
        #we find our max hypothetical q value for the next action by querying the table for this state and returning the maximum value
        maxQ = np.max(self.q_table[next_state_tuple]) #If this throws an error we can use python's dict.get feature to set a default if this fails to jsut set to zeros(12), but that shouldve been handled previously when we call
        #self._init_q_table_row_if_not(next_state_tuple) since that should init we dont need to use dict.get I think. 
        
        #apply the q learning formula to find new q value
        q = (1-a)*c + (a*(r+(L*maxQ)))
        
        #and now we update the table
        self.q_table[state_tuple][action_after_mapping] = q
        
        #so now when we query the table for the next action it has a better vibe of what is good or bad and we can choose based on that (assuming we arnt exploring)
        
    def q_learning_do_one_step(self,state):
        """a function that will step forward once in accordance with q learning's methodology and then update the tables
        
        args:
            you must provide state
            i dont think anything else
            
        returns:
            (boolean,dict): first value is if this epoch is completed, second is new state
        """
        #find our act
        act = self.find_next_action_pls(state)
        #take a step 
        next_state, reward, done = self.step(act)
        if(verbose): print(f"step {self.steps}: state:{state},reward:{reward},done:{done}")
        #update qlearning table
        self.q_learning_handler(state,act,reward,next_state)
        return (done,next_state)
    
    def train(self,epochs=100, predefined_state=None):
        """train the model on various random states

        Args:
            epochs (int, optional): how many epochs to train for. Defaults to 100.
        """
        global verbose
        prev_verbose = verbose
        if(verbose_in_training): verbose = verbose
        else: verbose = False
        print(f"training for {epochs} epochs")
        
        #reset to baseline, then loop solving problems for a while till enough epochs have passed
        state, _ = self.reset(predefined_state)
        for epoch in range(0,epochs):
            state, _ = self.reset(predefined_state)
            done = False
            while not done:
                done, state = self.q_learning_do_one_step(state)
        print(f"training done! {epochs} epochs trained for!")
        verbose = prev_verbose
    
    def solve(self,inital_state, maximum_allowed_steps=False):
        """solve a particular state using the trained q-learning model

        Args:
            state (_type_): the state to solve
        """
        prev_max_steps = self.max_steps
        self.max_steps = maximum_allowed_steps if maximum_allowed_steps else self.max_steps
        
        if(verbose): print(f"initial state provided: {inital_state}")
        if(verbose): print("solving based on q-table")
        state, _ = self.reset(inital_state)
        done = False
        while not done:
            done, state = self.q_learning_do_one_step(state)
        if(verbose): print(f"solving done")
        if(verbose): print(f"final state: {state}")
        
        self.max_steps = prev_max_steps
    
        
        
        return state
        

def train_test_isosceles(train=100,test=100,test_max_steps=1000):
    """train and test the rl model on solving only isosceles triangles
    train is how many triangle examples to train on
    test is how many examples to test on
    """

    mean_squared_error = train_test_generic(train=train,test=test, test_max_steps=test_max_steps, test_state_function=TriangleRL_Environment.iso_lambda, train_state_function=TriangleRL_Environment.iso_lambda)
    
    print(f"Mean Squared Error For isosceles (ASA) trained on isosceles (ASA): {mean_squared_error}")

def train_test_right_triangles(train=100,test=100,test_max_steps=1000):
    """train and test the rl model on solving only isosceles triangles
    train is how many triangle examples to train on
    test is how many examples to test on
    """

    mean_squared_error = train_test_generic(train=train,test=test, test_max_steps=test_max_steps, test_state_function=TriangleRL_Environment.right_angle_lambda, train_state_function=TriangleRL_Environment.right_angle_lambda)
    
    print(f"Mean Squared Error For Right Triangles trained on Right Triangles: {mean_squared_error}")
    
def train_rand_test_isosceles(train=100,test=100,test_max_steps=1000):
    """train and test the rl model on solving only isosceles triangles
    train is how many triangle examples to train on
    test is how many examples to test on
    """

    
    mean_squared_error = train_test_generic(train=train,test=test, test_max_steps=test_max_steps, test_state_function=TriangleRL_Environment.iso_lambda)
    
    print(f"Mean Squared Error For isosceles (ASA) trained on random (random): {mean_squared_error}")
    
def train_SAS_test_SAS(train=100,test=100,test_max_steps=1000):
    """train and test the rl model on solving only isosceles triangles
    train is how many triangle examples to train on
    test is how many examples to test on
    """

    mean_squared_error = train_test_generic(train=train,test=test,test_max_steps=test_max_steps, test_state_function=TriangleRL_Environment.iso_lambda,train_state_function=TriangleRL_Environment.iso_lambda)
    
    print(f"Mean Squared Error For SAS trained on SAS: {mean_squared_error}")
    
def train_Random_test_SAS(train=100,test=100,test_max_steps=1000):
    """train and test the rl model on solving only isosceles triangles
    train is how many triangle examples to train on
    test is how many examples to test on
    """

    mean_squared_error = train_test_generic(train=train,test=test,test_max_steps=test_max_steps, test_state_function=TriangleRL_Environment.iso_lambda)
    
    print(f"Mean Squared Error For SAS trained on Random: {mean_squared_error}")
    
def train_random_test_random(train=100,test=100,test_max_steps=1000):
    """train and test the rl model on solving random triangles of any type.
    """

    mean_squared_error = train_test_generic(train=train,test=test,test_max_steps=test_max_steps)
    
    print(f"Mean Squared Error For Random (Random) trained on Random (Random): {mean_squared_error}")
    
def train_test_generic(train_state_function=None,test_state_function=None,train=100,test=100,test_max_steps=1000):
    """a simple function that handles what types of triangles to train on, to test on in a modular way

    Args:
        train_state_function (_type_, optional): _description_. Defaults to None.
        test_state_function (_type_, optional): _description_. Defaults to None.
        train (int, optional): _description_. Defaults to 100.
        test (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    #first we need to craete our RL enviornment
    env = TriangleRL_Environment()
    
    #okay now we need to train the env
    train_state = train_state_function() if callable(train_state_function) else None
    env.train(train, train_state)
    
    #now lets try to solve a triangle, a equilateral triangle will have all sides of the same length, as such it should find [60,60,60,5,5,5]
    solved_states = []
    errorArray = []
    
    #find 100 sample problems to solve
    for n in range(test):
        test_state = test_state_function() if callable(test_state_function) else None
        state = env.solve(test_state,test_max_steps)
        solved_states.append(state)
        errorArray.append(env.real_triangle.get_error(state))
    print(errorArray)
    print(solved_states)
    
    #calculate the error of each of these, since they are isosolises in this case its just going to be how far off the sides are from the 1 provided side

    

    mean_squared_error = (sum([error**2 for error in errorArray])/len(solved_states))
    return mean_squared_error
    


def main():
    """the driver function to demo this file"""
    
    train=100
    test=100
    #train_test_right_triangles(train,test)
    train_test_isosceles(train,test,1000)
    #train_rand_test_isosceles(train,test)
    #train_random_test_random(train,test)
    #train_Random_test_SAS(train,test)
    #train_SAS_test_SAS(train,test)
    
            

if __name__ == "__main__":
    main()  
        
        
        
        

        
        