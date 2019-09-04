import numpy as np
import supper

controller = supper.AiController()
for i in range(100):
    internal_data = np.array([i,i,i,i,i,i])
    output_data = np.array([i,i])
    controller.append_internal_data(internal_data)
    controller.append_output_data(output_data)
    input_data = np.array([i,i])
    controller.append_input_data(input_data)
