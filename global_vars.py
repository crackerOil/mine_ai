open_new_plot_window = True

num_wins = 0

input_number = 24
output_number = 2

c_param_1 = 1
c_param_2 = 1
c_param_3 = 0.4
distance_threshold = 3

population_size = 20

global_innovs = {}

def new_innovation(input, output):
	global_innovs[(input, output)] = len(global_innovs) + 1