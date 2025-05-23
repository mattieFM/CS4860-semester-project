import math
from rl_model_2d_prediction import RL_2D_Predict
import random
random.seed(42)

verbose = False

def generic_driver(test_count=20,train_count=80,function_to_predict=math.sin, smallest_value_allowed=0.00001, base_increment_size=.001, label="generic"):
    #now lets try to solve a triangle, a equilateral triangle will have all sides of the same length, as such it should find [60,60,60,5,5,5]
    solved_states = []
    errorArray = []
    train_count = train_count
    test_count = test_count
    env = RL_2D_Predict(function_to_predict=function_to_predict, smallest_value_allowed=smallest_value_allowed, base_increment_size=base_increment_size)
    env.train(train_count)
    
    #find 100 sample problems to solve
    for n in range(test_count):
        state = env.solve(None)
        solved_states.append(state)
        errorArray.append(env.get_error(state[0],state[1]))
        
    if(verbose):
        print(errorArray)
        print(solved_states)
    
    #calculate the error of each of these, since they are isosolises in this case its just going to be how far off the sides are from the 1 provided side

    mean_squared_error = (sum([error**2 for error in errorArray])/len(solved_states))
    
    print(f"MSE for predicting '{label}':{mean_squared_error}")

def main():
    
    train = 80
    test = 20
    print(f"80/20")
    generic_driver(test,train,math.sin,0.00001,0.001,"sin")
    generic_driver(test,train,math.cos,0.00001,0.001,"cos")
    generic_driver(test,train,math.tan,0.00001,0.001,"tan")
    
    
    train = 800
    test = 200
    print(f"800/200")
    generic_driver(test,train,math.sin,0.00001,0.001,"sin")
    generic_driver(test,train,math.cos,0.00001,0.001,"cos")
    generic_driver(test,train,math.tan,0.00001,0.001,"tan")
    
    train = 1600
    test = 400
    print(f"1600/400")
    generic_driver(test,train,math.sin,0.00001,0.001,"sin")
    generic_driver(test,train,math.cos,0.00001,0.001,"cos")
    generic_driver(test,train,math.tan,0.00001,0.001,"tan")
    
if __name__ == "__main__":
    main()  
        