import numpy as np
import win32api, win32con
import pyscreenshot
import time
import operator
import random
import math
import networkx as nx
import matplotlib.pyplot as plt

from reading_board import GetInput
from clicking_tiles import GetOutput

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
		
class Gene():
	def __init__(self, in_node, out_node, weight, enabled, innov):
		self.in_node = in_node
		self.out_node = out_node
		self.weight = weight
		self.enabled = enabled
		self.innov = innov
		
class Genome():
	def __init__(self):
		self.genotype = []
		self.network = {}
		self.network_layers = 0
		self.fitness = 0
		self.adjusted_fitness = 0
	
	def new_gene(self, in_node, out_node, weight, enabled):
		if not ((in_node, out_node) in global_innovs):
			new_innovation(in_node, out_node)
			
		innov = global_innovs[(in_node, out_node)]
			
		self.genotype.append(Gene(in_node, out_node, weight, enabled, innov))
	
	def print_genome(self):
		for gene in self.genotype:
			print("\nGene " + str(gene.innov) + ": " + str(gene.in_node) + " -> " + str(gene.out_node) + " (" + str(gene.weight) + ") " + str(gene.enabled))
		
class Neuron():
	def __init__(self):
		self.inputs = []
		self.value = 0
		self.layer = None
		
def new_network(genome):
	network = {}
	
	for i in range(1, (input_number + output_number + 1)):
		network[i] = Neuron()
		network[i].layer = 0
		
	for j in range(0, len(genome.genotype)):
		gene = genome.genotype[j]
		
		if gene.enabled:
			if not (gene.out_node in network):
				network[gene.out_node] = Neuron()
				
			network[gene.out_node].inputs.append(gene)
			
			if not (gene.in_node in network):
				network[gene.in_node] = Neuron()
	
	not_checked_nodes = [node for node in range((input_number + 1), (input_number + output_number + 1))]
	while len(not_checked_nodes) != 0:
		active_node = not_checked_nodes[0]
		
		for gene in network[active_node].inputs:
			if gene.in_node > input_number:
				network[gene.in_node].layer = network[active_node].layer + 1
			
				not_checked_nodes.append(gene.in_node)
			
				genome.network_layers = max(genome.network_layers, network[gene.in_node].layer)
			
		not_checked_nodes = not_checked_nodes[1:]
				
	genome.network = network
	
def feed_network(network, inputs):
	outputs = []

	for i in range(1, (input_number + 1)):
		network[i].value = inputs[i - 1]
		
	for j in list(network.keys()):
		if j > (input_number + output_number):
			neuron = network[j]
	
			sum = 0
			for gene in neuron.inputs:
				sum += gene.weight * network[gene.in_node].value
			
			sum = 1 / (1 + math.exp(-4.9 * sum)) # Modified sigmoid
		
			neuron.value = sum
		
	for k in range((input_number + 1), (input_number + output_number + 1)):
		neuron = network[k]
	
		sum = 0
		for gene in neuron.inputs:
			sum += gene.weight * network[gene.in_node].value
			
		sum = 1 / (1 + math.exp(-4.9 * sum)) # Modified sigmoid
		
		neuron.value = sum
		
		outputs.append(neuron.value)
		
	return outputs

def show_network(genome):
	# directed graph
	network_graph = nx.DiGraph()
	
	sorted_network_nodes = sorted(list(genome.network.keys()), reverse=False)
	
	# calculate number of nodes in each layer
	layer_pops = {}
	for node in sorted_network_nodes:
		try:
			layer_pops[genome.network[node].layer] += 1
		except:
			layer_pops[genome.network[node].layer] = 1
	
	# set up positions of each node on graph based on how many in each layer and how many layers
	for node in sorted_network_nodes:
		node_pos = None
		
		num_in_layer = 1
		
		if node <= input_number:
			node_pos = (0.05, node * (1 / (input_number + 1)))
		elif node <= (input_number + output_number):
			node_pos = (0.95, (node - input_number) * (1 / (output_number + 1)))
		else:
			node_pos = (1 - (genome.network[node].layer * (0.9 / (genome.network_layers + 1)) + 0.05), num_in_layer * (1 / (layer_pops[genome.network[node].layer] + 1)))
			
			if num_in_layer == layer_pops[genome.network[node].layer]:
				num_in_layer = 1
			else:
				num_in_layer += 1
				
		network_graph.add_node(node, pos=node_pos)
	
	# set up edges of graph
	for edge in genome.genotype:
		if edge.enabled:
			network_graph.add_edge(edge.in_node, edge.out_node, weight=round(edge.weight, 2))

	pos = nx.get_node_attributes(network_graph, "pos")
	nx.draw(network_graph, pos, node_size=75, with_labels=False)
	labels = nx.get_edge_attributes(network_graph,"weight")
	nx.draw_networkx_edge_labels(network_graph, pos, edge_labels=labels)
	
	# change position and size of window and make sure it doesn't block further computation
	#plt.draw()
	if open_new_plot_window:
		plt.ion()
		fig = plt.gcf()
		mngr = plt.get_current_fig_manager()
		mngr.window.wm_geometry("+%d+%d" % (951, 0))
		fig.set_size_inches(9.6, 5)
	plt.show()
	plt.pause(0.001)

def close_network():
	plt.clf()	
	
def calc_compatibility_dist(individual_1, individual_2, c1, c2, c3):
	sorted_individual_1 = sorted(individual_1.genotype, key=lambda x: x.innov, reverse=False)
	sorted_individual_2 = sorted(individual_2.genotype, key=lambda x: x.innov, reverse=False)
			
	matching_genes = 0
	disjoints = 0
	excesses = 0
	avg_weight_dist = 0
	len_bigger_genome = max(len(sorted_individual_1), len(sorted_individual_2))
	
	i = 0
	while i < len_bigger_genome:
		try:
			if sorted_individual_1[i].innov != sorted_individual_2[i].innov:
				disjoints += 1
				
				if sorted_individual_1[i].innov > sorted_individual_2[i].innov:
					sorted_individual_1.insert(i, Gene(None, None, None, None, None))
				else:
					sorted_individual_2.insert(i, Gene(None, None, None, None, None))
			else:
				matching_genes += 1
				avg_weight_dist += sorted_individual_1[i].weight - sorted_individual_2[i].weight
		except:
			excesses += 1
			
		len_bigger_genome = max(len(sorted_individual_1), len(sorted_individual_2))
		i += 1
		
	try:		
		avg_weight_dist = abs(avg_weight_dist / matching_genes)
	except:
		avg_weight_dist = 1000
			
	compatibility_dist = ((c1 * excesses) / len_bigger_genome) + ((c2 * disjoints) / len_bigger_genome) + (c3 * avg_weight_dist)
	
	#print("\nIndividual 1:")
	#for gene_1 in sorted_individual_1:
	#		print("\nGene " + str(gene_1.innov) + ": " + str(gene_1.in_node) + " -> " + str(gene_1.out_node) + " (" + str(gene_1.weight) + ") " + str(gene_1.enabled))
	
	#print("\nIndividual 2:")	
	#for gene_2 in sorted_individual_2:
	#		print("\nGene " + str(gene_2.innov) + ": " + str(gene_2.in_node) + " -> " + str(gene_2.out_node) + " (" + str(gene_2.weight) + ") " + str(gene_2.enabled))
			
	#print("\nDisjoints: " + str(disjoints))
	#print("\nExcesses: " + str(excesses))
	#print("\nCompatibility dist: " + str(compatibility_dist))
	
	return compatibility_dist
	
def speciation(population, species_list, c1, c2, c3, threshold):
	for individual in population:
		best_compatibility = {"species": None, "score": 100}
		
		for j in list(species_list.keys()):
			compatibility_dist = calc_compatibility_dist(species_list[j]["individuals"][0], individual, c1, c2, c3)
			
			if compatibility_dist < best_compatibility["score"]:
				best_compatibility["species"] = j
				best_compatibility["score"] = compatibility_dist
				
		if best_compatibility["score"] <= threshold:
			species_list[best_compatibility["species"]]["individuals"].append(individual)
		else:
			species_list[max(list(species_list.keys())) + 1] = {"individuals": [individual],
																"max_fitness": 0,
																"gens_without_fitness_improvement": 0
															   }
			
def adjust_fitness(species_list, c1, c2, c3, threshold):
	for i in list(species_list.keys()):
		for individual in species_list[i]["individuals"]:
			sharing_sum = 0.5
			
			for neighbour in species_list[i]["individuals"]:
				if individual != neighbour:
					if calc_compatibility_dist(individual, neighbour, c1, c2, c3) <= threshold:
						sharing_sum += 1
			
			individual.adjusted_fitness = individual.fitness / sharing_sum
			
def crossover(genome_1, genome_2):
	child = Genome()

	parent_1 = {}
	parent_2 = {}
	
	for gene in genome_1.genotype:
		parent_1[gene.innov] = gene
	
	for gene in genome_2.genotype:
		parent_2[gene.innov] = gene
		
	if genome_1.adjusted_fitness > genome_2.adjusted_fitness:
		fitter_parent = parent_1
		lesser_parent = parent_2
	else:
		fitter_parent = parent_2
		lesser_parent = parent_1
		
	for allele in fitter_parent:
		try:
			allele_2 = lesser_parent[allele]
			
			if random.randint(1, 2) == 1:
				child.genotype.append(fitter_parent[allele])
			else:
				child.genotype.append(allele_2)
		except:
			child.genotype.append(fitter_parent[allele])
			
	return child

def new_innovation(input, output):
	global_innovs[(input, output)] = len(global_innovs) + 1
	
def mutate_genome(genome, mutation_rate): # mutation_rate ~ 0.3 maybe?
	new_network(genome)

	if (random.randint(1, 100) / 100) <= mutation_rate:
		choice = random.randint(1, 100)
		if choice <= 5: # mutate_link
			#print("link")
			if len(genome.network) ** 2 - input_number * len(genome.network) - output_number * len(genome.network) < len(genome.genotype):
				mutate_genome(genome, 1)
			else:
				in_node_good = False
				while not in_node_good:
					in_node = random.randint(1, len(genome.network))
				
					if not ((in_node > input_number) and (in_node <= input_number + output_number)):
						in_node_good = True
				
				out_node_good = False
				while not out_node_good:
					out_node = random.randint(input_number + 1, len(genome.network))
				
					if in_node != out_node:
						out_node_good = True
				
				in_genome_already = False
				
				if (in_node, out_node) in global_innovs:
					for i in range(0, len(genome.genotype)):
						if genome.genotype[i].innov == global_innovs[(in_node, out_node)]:
							in_genome_already = True
				
				if not in_genome_already:
					genome.new_gene(in_node, out_node, random.uniform(-2, 2), True)
				else:
					mutate_genome(genome, 1)
		elif choice <= 8: # mutate_node
			#print("node")
			yes = True
			while yes:
				node_pos = random.randint(0, len(genome.genotype) - 1)
				if genome.genotype[node_pos].enabled:
					yes = False
			
			genome.genotype[node_pos].enabled = False
			
			genome.new_gene(genome.genotype[node_pos].in_node, len(genome.network) + 1, 1, True)

			genome.new_gene(len(genome.network) + 1, genome.genotype[node_pos].out_node, genome.genotype[node_pos].weight, True)
		elif choice <= 20: # mutate_enable_disable
			#print("ena/dis")
			if len(genome.genotype) == 1:
				mutate_genome(genome, 1)
			else:
				connection = random.randint(0, len(genome.genotype) - 1)
			
				genome.genotype[connection].enabled = not genome.genotype[connection].enabled
		elif choice <= 92: # mutate_weight_shift
			#print("shift")
			connection = random.randint(0, len(genome.genotype) - 1)
			
			genome.genotype[connection].weight = genome.genotype[connection].weight * random.uniform(0, 2)
		else: # mutate_weight_random
			#print("rand")
			connection = random.randint(0, len(genome.genotype) - 1)
			
			genome.genotype[connection].weight = random.uniform(-2, 2)
			
def selection(population, species_list, c1, c2, c3, threshold, size_of_pop, generation):
	speciation(population, species_list, c1, c2, c3, threshold)
	
	if generation == 0:
		# remove first element left over from initial population
		species_list[1]["individuals"] = species_list[1]["individuals"][1:]
	else:
		for l in list(species_list.keys()):
			# remove first element of each species left over from last speciation
			species_list[l]["individuals"] = species_list[l]["individuals"][1:]
			
			# delete unpopulated species
			if len(species_list[l]["individuals"]) == 0:
				del species_list[l]
	
	adjust_fitness(species_list, c1, c2, c3, threshold)

	# for every species, sort by fitness and purge 50%
	for i in list(species_list.keys()):
		species_list[i]["individuals"].sort(key=lambda x: x.adjusted_fitness, reverse=True)
		
		species_list[i]["individuals"] = species_list[i]["individuals"][:math.ceil(len(species_list[i]["individuals"]) / 2)]
	
	new_population = []
	
	for j in list(species_list.keys()):
		# best individual in species with more than 5 members copied onto next generation
		if len(species_list[j]["individuals"]) > 5:
			new_population.append(species_list[j]["individuals"][0])
		
		# if the max fitness in the species is greater than the last recorded max
		if max(species_list[j]["individuals"], key=lambda x: x.adjusted_fitness).adjusted_fitness > species_list[j]["max_fitness"]:
			# update the max
			species_list[j]["max_fitness"] = max(species_list[j]["individuals"], key=lambda x: x.adjusted_fitness).adjusted_fitness
		else:
			# otherwise, another useless generation
			species_list[j]["gens_without_fitness_improvement"] += 1
	
	while len(new_population) < size_of_pop:
		mating_species = species_list[random.choice(list(species_list.keys()))]
		
		if mating_species["gens_without_fitness_improvement"] < 15:
			first_parent = random.choice(mating_species["individuals"])
		
			if random.randint(1, 1000) == 1: # 0.1% chance of interspecies mating
				# although could still yield intraspecies mating, failsafe in case only 1 species alive
				second_parent = random.choice(species_list[random.choice(list(species_list.keys()))]["individuals"])
			else:
				# might asexually reproduce but failsafe in case of 1 individual in species
				second_parent = random.choice(mating_species["individuals"])
			
			crossover_child = crossover(first_parent, second_parent)
			
			# 30% mutation chance
			mutate_genome(crossover_child, 0.3)
			
			new_population.append(crossover_child)
			
	for k in list(species_list.keys()):
		# preserve only first individual of species for next gen speciation comparisons
		species_list[k]["individuals"] = [species_list[k]["individuals"][0]]
			
	return new_population

def generate_initial_pop(size):
	population = []
	
	for i in range(0, size):
		new_genome = Genome()
		
		in_node = random.randint(1, input_number)

		out_node = random.randint(input_number + 1, input_number + output_number)
		
		new_genome.new_gene(in_node, out_node, random.uniform(-2, 2), True)
		
		new_network(new_genome)
		
		population.append(new_genome)
		
	return population
		
			
def play_individual(genome):
	new_network(genome)

	show_network(genome)
	
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
							
					outputs = feed_network(genome.network, inputs)
				
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
	
	close_network()
	
def train(initial_population, c1, c2, c3, threshold, size_of_pop):
	population = initial_population
	
	generation = 0
	
	species_list = {1: {
						"individuals": [population[0]], 
						"max_fitness": 0, 
						"gens_without_fitness_improvement": 0
					   }
				   }
	while generation < 50:
		print("\nGeneration " + str(generation))
		
		if generation != 0:
			last_gen_best_fit = max(list(species_list.keys()), key=lambda x: species_list[x]["max_fitness"])
			print("\nLast generation best fitness: " + str(species_list[last_gen_best_fit]["max_fitness"]) + " (Species " + str(last_gen_best_fit) + ")")
		
			print("\n")
			species_list[last_gen_best_fit]["individuals"][0].print_genome()
			print("\n")
		
		for individual in population:
			individual.print_genome()
			play_individual(individual)
			
			open_new_plot_window = False
		
		population = selection(population, species_list, c1, c2, c3, threshold, size_of_pop, generation)
		
		generation += 1
	
	
#genome_1 = Genome()
#genome_1.new_gene(1, 4, 0.7, True) # 1
#genome_1.new_gene(2, 4, -0.5, False) # 2
#genome_1.new_gene(3, 4, 0.5, True) # 3
#genome_1.new_gene(2, 5, 0.2, True) # 4
#genome_1.new_gene(5, 4, 0.4, True) # 5
#genome_1.adjusted_fitness = 100

#genome_2 = Genome()
#genome_2.new_gene(1, 4, 0.2, True) # 1
#genome_2.new_gene(2, 4, -0.7, False) # 2
#genome_2.new_gene(3, 4, 0.1, True) # 3
#genome_2.new_gene(2, 5, 1.3, True) # 4
#genome_2.new_gene(5, 6, 0.3, True) # 6
#genome_2.new_gene(6, 4, -0.2, True) # 7
#genome_2.new_gene(3, 5, 0.9, False) # 8
#genome_2.new_gene(1, 6, 0.1, True) # 9
#genome_2.adjusted_fitness = 150

#genome_3 = crossover(genome_1, genome_2)

#new_network(genome_1)

#outputs = feed_network(genome_3.network, [1, 1, 1])

#print(outputs)
#print(genome_3.network)

#genome_1.print_genome()

#mutate_genome(genome_1, 1)

#genome_1.print_genome()

#pop[0].print_genome()

#input()

#play_individual(pop[0])

#print(pop[0].fitness)

#for creature in pop:
#	creature.print_genome()

#species_list = {1: {"individuals": [pop[0]], "max_fitness": 0, "gens_without_fitness_improvement": 0}}

#speciation(pop, species_list, 1, 1, 0.4, 3)

#for m in list(species_list.keys()):
#	print("\nSpecies " + str(m) + ":")
#	for cre in species_list[m]["individuals"]:
#		cre.print_genome()

#calc_compatibility_dist(genome_1, genome_2, 1, 1, 0.4)

#new_pop = selection(pop, species_list, c_param_1, c_param_2, c_param_3, distance_threshold, population_size)

#print(new_pop)

#new_network(pop[0])

#show_network(pop[0])

#input()

#close_network()

#species_list = {1: {
#						"individuals": [pop[0]], 
#						"max_fitness": 0, 
#						"gens_without_fitness_improvement": 0
#					   }
#				   }
				   
#pop = selection(pop, species_list, c_param_1, c_param_2, c_param_3, distance_threshold, population_size, 0)

#print(pop)
#print(species_list)

pop = generate_initial_pop(population_size)

input_data = GetInput("intermediate")
output_data = GetOutput("intermediate")

train(pop, c_param_1, c_param_2, c_param_3, distance_threshold, population_size)

input()