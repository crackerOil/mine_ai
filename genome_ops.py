import math

import networkx as nx
import matplotlib.pyplot as plt

from global_vars import *

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
	
	def build_network(self):
		network = {}
	
		for i in range(1, (input_number + output_number + 1)):
			network[i] = Neuron()
			network[i].layer = 0
		
		for j in range(0, len(self.genotype)):
			gene = self.genotype[j]
		
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
			
					self.network_layers = max(self.network_layers, network[gene.in_node].layer)
			
			not_checked_nodes = not_checked_nodes[1:]
				
		self.network = network
		
	def feed_network(self, inputs):
		outputs = []

		for i in range(1, (input_number + 1)):
			self.network[i].value = inputs[i - 1]
		
		for j in list(self.network.keys()):
			if j > (input_number + output_number):
				neuron = self.network[j]
	
				sum = 0
				for gene in neuron.inputs:
					sum += gene.weight * self.network[gene.in_node].value
			
				sum = 1 / (1 + math.exp(-4.9 * sum)) # Modified sigmoid
		
				neuron.value = sum
		
		for k in range((input_number + 1), (input_number + output_number + 1)):
			neuron = self.network[k]
	
			sum = 0
			for gene in neuron.inputs:
				sum += gene.weight * self.network[gene.in_node].value
			
			sum = 1 / (1 + math.exp(-4.9 * sum)) # Modified sigmoid
		
			neuron.value = sum
		
			outputs.append(neuron.value)
		
		return outputs
		
	def show_network(self, new_window):
		# directed graph
		network_graph = nx.DiGraph()
	
		sorted_network_nodes = sorted(list(self.network.keys()), reverse=False)
	
		# calculate number of nodes in each layer
		layer_pops = {}
		for node in sorted_network_nodes:
			try:
				layer_pops[self.network[node].layer] += 1
			except:
				layer_pops[self.network[node].layer] = 1
	
		# set up positions of each node on graph based on how many in each layer and how many layers
		for node in sorted_network_nodes:
			node_pos = None
		
			num_in_layer = 1
		
			if node <= input_number:
				node_pos = (0.05, node * (1 / (input_number + 1)))
			elif node <= (input_number + output_number):
				node_pos = (0.95, (node - input_number) * (1 / (output_number + 1)))
			else:
				node_pos = (1 - (self.network[node].layer * (0.9 / (self.network_layers + 1)) + 0.05), num_in_layer * (1 / (layer_pops[genome.network[node].layer] + 1)))
			
				if num_in_layer == layer_pops[self.network[node].layer]:
					num_in_layer = 1
				else:
					num_in_layer += 1
				
			network_graph.add_node(node, pos=node_pos)
	
		# set up edges of graph
		for edge in self.genotype:
			if edge.enabled:
				network_graph.add_edge(edge.in_node, edge.out_node, weight=round(edge.weight, 2))

		pos = nx.get_node_attributes(network_graph, "pos")
		nx.draw(network_graph, pos, node_size=75, with_labels=False)
		labels = nx.get_edge_attributes(network_graph,"weight")
		nx.draw_networkx_edge_labels(network_graph, pos, edge_labels=labels)
	
		# change position and size of window and make sure it doesn't block further computation
		#plt.draw()
		if new_window:
			plt.ion()
			fig = plt.gcf()
			mngr = plt.get_current_fig_manager()
			mngr.window.wm_geometry("+%d+%d" % (951, 0))
			fig.set_size_inches(9.6, 5)
		plt.show()
		plt.pause(0.001)

	def close_network(self):
		plt.clf()
	
class Neuron():
	def __init__(self):
		self.inputs = []
		self.value = 0
		self.layer = None