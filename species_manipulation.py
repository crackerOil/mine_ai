import numpy as np
import operator
import random
import math

from global_vars import *

from reading_board import *
from clicking_tiles import *

from genome_ops import *

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
	
def mutate_genome(genome, mutation_rate): # mutation_rate ~ 0.3 maybe?
	genome.build_network()

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