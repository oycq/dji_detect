import time
import numpy as np
import supper
import random
controller = supper.AiController()

j = random.gauss(0,1)
input_data = np.array([0.25,0.25])
for i in range(15000):
#    j = (i % 200 - 100) / 100# + random.gauss(0,0.002)
    internal_data = np.array([0,1,0,0,0])
    z = input_data[0]
    z = (z*z/2 - 2)*(z-1)
    output_data = [z,z]
    controller.append_internal_data(internal_data)
    controller.append_output_data(output_data)
#    if i == 22:
#        print(controller.database.get_train_batch()['prior'])
#        print(controller.database.get_train_batch()['input'])
#        print(controller.database.get_train_batch()['output'])
#        print(controller.database.get_current_prior_info())

    asd = time.time() * 1000
    j = random.gauss(0,1)
    input_data = np.array([j,j])
    if i  >= 21:
        controller.train()
    #    print(time.time()*1000-asd)
        a = controller.get_optimized_control()
    #    print(time.time()*1000-asd)
        input_data = a
    controller.append_input_data(input_data)
    #print(time.time()*1000-asd)
