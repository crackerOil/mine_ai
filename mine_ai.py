import numpy as np
import time
import operator
import random

from global_vars import *

from reading_board import *
from clicking_tiles import *

from genome_ops import *

from species_manipulation import *

def generate_initial_pop(size):
	population = []
	
	for i in range(0, size):
		new_genome = Genome()
		
		in_node = random.randint(1, input_number)

		out_node = random.randint(input_number + 1, input_number + output_number)
		
		new_genome.new_gene(in_node, out_node, random.uniform(-2, 2), True)
		
		new_genome.build_network()
		
		population.append(new_genome)
		
	return population
		
			
def play_individual(genome):
	genome.build_network()

	genome.show_network(open_new_plot_window)
	
	time.sleep(3)
	
	game_lost = False
	game_won = False
	
	prev_board = np.zeros((input_data.rows, input_data.cols))
	
	num_flags = 0 # in case it surpases num of bombs, get loss
	
	output_data.left_click(int(input_data.cols/2), int(input_data.rows/2))
	
	time.sleep(0.05)
	
	while (not game_lost) and (not game_won):
		board = input_data.read_board()
		
		best_choice = {"x": None,
					   "y": None,
					   "score": -1,
					   "operation": None
					  }
	
		for nx in range(0, input_data.cols):
			for ny in range(0, input_data.rows):
				if board[ny, nx] == 0:
					inputs = []
			
					for i in range(-1, 2):
						for j in range(-1, 2):
							if i == 0 and j == 0:
								pass
							else:
								try:
									inputs.append(board[ny + i][nx + j])
								except:
									inputs.append(0.5)
							
								empty_sum = 0
								flag_sum = 0
							
								for k in range(-1, 2):
									for l in range(-1, 2):
										if k == 0 and l == 0:
											pass
										else:
											try:
												if board[ny + i + k][nx + j + l] == 0:
													empty_sum += 1
												elif board[ny + i + k][nx + j + l] == -1:
													flag_sum += 1
											except:
												pass
							
								inputs.append(empty_sum)
								inputs.append(flag_sum)
							
					outputs = genome.feed_network(inputs)
				
					if outputs[0] > outputs[1]:
						if best_choice["score"] < outputs[0]:
							best_choice["x"] = nx
							best_choice["y"] = ny
							best_choice["score"] = outputs[0]
							best_choice["operation"] = "open"
					else:
						if best_choice["score"] < outputs[1]:
							best_choice["x"] = nx
							best_choice["y"] = ny
							best_choice["score"] = outputs[1]
							best_choice["operation"] = "flag"
				elif board[ny, nx] == -100:
					game_lost = True
		
		time.sleep(0.05)
		
		if num_flags > input_data.mines:
			game_lost = True
		
		if game_lost:
			for fx in range(0, input_data.cols):
				for fy in range(0, input_data.rows):
					if prev_board[fy][fx] == -1:
						if board[fy][fx] == -100:
							genome.fitness += 5
						else:
							genome.fitness -= 5
			
			print("Game lost with raw fitness " + str(genome.fitness) + ". Moving on to next individual... ")
		elif game_won:
			genome.fitness += 100
		
			print("Game WON with raw fitness " + str(genome.fitness) + ". Moving on to next individual... ")
			
			#input()
		else:
			if best_choice["operation"] == "open":
				output_data.left_click(best_choice["x"], best_choice["y"])
				
				time.sleep(0.1)
				
				genome.fitness += 5
			elif best_choice["operation"] == "flag":
				output_data.right_click(best_choice["x"], best_choice["y"])
				
				time.sleep(0.1)
				
				num_flags += 1
			else:
				print("Something went wrong! (line 621)")
				print("Retrying... ")
				
				time.sleep(5)
				
			prev_board = board
	
	output_data.reset_board()
	
	time.sleep(0.05)
	
	genome.close_network()
	
def train(initial_population, c1, c2, c3, threshold, size_of_pop):
	population = initial_population
	
	generation = 0
	
	species_list = {1: {
						"individuals": [population[0]], 
						"max_fitness": 0, 
						"gens_without_fitness_improvement": 0
					   }
				   }
	while generation < 100:	
		if generation != 0:
			last_gen_best_fit = max(list(species_list.keys()), key=lambda x: species_list[x]["max_fitness"])
			print("\nLast generation best fitness: " + str(species_list[last_gen_best_fit]["max_fitness"]) + " (Species " + str(last_gen_best_fit) + ")")
			species_list[last_gen_best_fit]["individuals"][0].print_genome()
			print("\n")
		
		print("\nGeneration " + str(generation))
		
		for individual in population:
			individual.print_genome()
			play_individual(individual)
			
			open_new_plot_window = False
		
		population = selection(population, species_list, c1, c2, c3, threshold, size_of_pop, generation)
		
		generation += 1
		
	return population

pop = generate_initial_pop(population_size)

input_data = GetInput("intermediate")
output_data = GetOutput("intermediate")

continue_training = "y"
while continue_training == "y":
	pop = train(pop, c_param_1, c_param_2, c_param_3, distance_threshold, population_size)

	continue_training = input("Would you like to continue the training? (y or n)")

#TODO save best individual or save whole population	
input()