import random, time, pulp, matplotlib, math
import networkx as nx
import numpy as np
from pprint import pprint
from collections import OrderedDict
matplotlib.use('Agg')
import matplotlib.pyplot as plt
random.seed(1000)  # for replicability of experiments.
__author__ = 'q4fj4lj9'




def generate_instance():

	NO_OF_REQUESTS_WO_SIR = 4
	MAX_PERCENT_EXTRA_REQUESTS = 100

	GRID_MAX   = 10
	GRID_MAX_X = GRID_MAX
	GRID_MAX_Y = GRID_MAX

	MAX_NO_BETAS  = 2
	MAX_BETA = 1000
	MAX_SENSITIVITY = 1000

	OUR_CUT_FROM_DRIVER = 0.3
	GAMMA = 0
	ALPHA_OP = MAX_SENSITIVITY/2


	all_requests = OrderedDict()
	for i in range(int(NO_OF_REQUESTS_WO_SIR*(1+ 0.01*MAX_PERCENT_EXTRA_REQUESTS))):
		all_requests[i] = OrderedDict()
		all_requests[i]['orig'] = (0,0)
		all_requests[i]['dest'] = (0,0)
		if all_requests[i]['orig']==all_requests[i]['dest']:
			all_requests[i]['orig'] = np.array([random.randint(0,GRID_MAX_X),random.randint(0,GRID_MAX_Y)])
			all_requests[i]['dest'] = np.array([random.randint(0,GRID_MAX_X),random.randint(0,GRID_MAX_Y)])

		all_requests[i]['detour_sensitivity'] = random.randint(1,MAX_SENSITIVITY)

		all_requests[i]['detour_sensitivity_normalized'] = random.randint(1,MAX_SENSITIVITY)*1.0/ALPHA_OP

		all_requests[i]['prices_per_unit_dist'] = {}
		n = sorted(random.sample(range(1,MAX_BETA),MAX_NO_BETAS))
		#print n
		for k in range(1,MAX_NO_BETAS+1):
			all_requests[i]['prices_per_unit_dist'][k] = n[k-1]*1.0/MAX_BETA #k-1 because n is indexed from 0 instead of 1

	return {'all_requests': all_requests,
	'params':{
	'OUR_CUT_FROM_DRIVER': OUR_CUT_FROM_DRIVER,
	'NO_OF_REQUESTS_WO_SIR': NO_OF_REQUESTS_WO_SIR,
	'MAX_PERCENT_EXTRA_REQUESTS':MAX_PERCENT_EXTRA_REQUESTS,
	'GAMMA': GAMMA}}

def euclidean(x,y):
	assert x is not None and y is not None
	return np.linalg.norm(x - y)

def get_driving_distance(selected_requests,permutation,instance):

	#local
	all_requests = instance['all_requests'] #TODO
	params = instance['params']

	#Logic
	if len(selected_requests) > 2:
		return NotImplementedError

	if len(selected_requests) == 1: #redundant
		return euclidean(all_requests[selected_requests[0]]['orig'],all_requests[selected_requests[0]]['dest'])

	if len(selected_requests) == 2:

		i = selected_requests[0]
		j = selected_requests[1]	
		if permutation[0]=='sj':#i and j are symmetric within each case. See board photo. TODO
			(i,j) = (j,i)

		if all_permutations[permutation] == 'case1':
			
			return euclidean(all_requests[i]['orig'],all_requests[j]['orig']) + \
					euclidean(all_requests[j]['orig'],all_requests[i]['dest']) + \
					euclidean(all_requests[i]['dest'],all_requests[j]['dest'])

		elif all_permutations[permutation] == 'case2':

			return euclidean(all_requests[i]['orig'],all_requests[j]['orig']) + \
					euclidean(all_requests[j]['orig'],all_requests[j]['dest']) + \
					euclidean(all_requests[j]['dest'],all_requests[i]['dest'])
		else:
			print "Error!" #TODO
			return -1 #TODO

def check_SIR_satisfaction(selected_requests,permutation,instance,all_permutations):
	
	all_requests = instance['all_requests'] #TODO
	params = instance['params']

	#Logic
	if len(selected_requests) > 2:
		return NotImplementedError

	if len(selected_requests) == 1: #redundant
		return True

	if len(selected_requests) == 2:

		i = selected_requests[0]
		j = selected_requests[1]

		if permutation[0]=='sj':#i and j are symmetric within each case. See board photo. TODO
			(i,j) = (j,i)		

		if all_permutations[permutation] == 'case1':

			constraint_i =  (all_requests[i]['prices_per_unit_dist'][1] - all_requests[i]['prices_per_unit_dist'][2])*euclidean(all_requests[i]['orig'],all_requests[i]['dest']) \
				>= (params['GAMMA'] + all_requests[i]['detour_sensitivity_normalized'])*\
					(euclidean(all_requests[i]['orig'],all_requests[j]['orig']) + \
					euclidean(all_requests[j]['orig'],all_requests[i]['dest']) - \
					euclidean(all_requests[i]['orig'],all_requests[i]['dest']))
			constraint_j = (all_requests[j]['prices_per_unit_dist'][1] - all_requests[j]['prices_per_unit_dist'][2])*euclidean(all_requests[j]['orig'],all_requests[j]['dest']) \
				>= (params['GAMMA'] + all_requests[j]['detour_sensitivity_normalized'])*\
					(euclidean(all_requests[j]['orig'],all_requests[i]['dest']) + \
					euclidean(all_requests[i]['dest'],all_requests[j]['dest']) - \
					euclidean(all_requests[j]['orig'],all_requests[j]['dest']))

		elif all_permutations[permutation] == 'case2':

			constraint_i = (all_requests[i]['prices_per_unit_dist'][1] - all_requests[i]['prices_per_unit_dist'][2])*euclidean(all_requests[i]['orig'],all_requests[i]['dest']) \
				>= (params['GAMMA'] + all_requests[i]['detour_sensitivity_normalized'])*\
					(euclidean(all_requests[i]['orig'],all_requests[j]['orig']) + \
					euclidean(all_requests[j]['orig'],all_requests[j]['dest']) + \
					euclidean(all_requests[j]['dest'],all_requests[i]['dest']) - \
					euclidean(all_requests[i]['orig'],all_requests[i]['dest']))
			constraint_j = all_requests[j]['prices_per_unit_dist'][1] >= all_requests[j]['prices_per_unit_dist'][2]

		else:
			print "Error!" #TODO
			constraint_i = False
			constraint_j = False

		return constraint_i and constraint_j


def get_profit(selected_requests,instance):
	assert instance is not None
	assert selected_requests is not None

	#Local copies
	all_requests = instance['all_requests']
	params = instance['params']

	#Logic
	if len(selected_requests) > 2:
		return NotImplementedError

	if len(selected_requests) == 1:
		return [all_requests[selected_requests[0]]['prices_per_unit_dist'][1]*euclidean(all_requests[selected_requests[0]]['orig'],all_requests[selected_requests[0]]['dest']),('si','di')]

	if len(selected_requests) == 2:
		i = selected_requests[0]
		j = selected_requests[1]

		all_permutations = {('si','sj','di','dj'):'case1',
			('si','sj','dj','di'):'case2',
			('sj','si','dj','di'):'case1',
			('sj','si','di','dj'):'case2'}#TODO: make more comprehensible

		profits = np.zeros(len(all_permutations))
		max_profit = 0
		max_profit_permutation = None
		for k,permutation in enumerate(all_permutations):
			SIR_satisfied = check_SIR_satisfaction([i,j],permutation,instance, all_permutations)
			if SIR_satisfied is True:
				profits[k] = params['ALPHA_OP']*(sum([all_requests[m]['prices_per_unit_dist'][2]*euclidean(all_requests[m]['orig'],all_requests[m]['dest']) for m in (i,j)]) - (1-params['OUR_CUT_FROM_DRIVER'])*get_driving_distance([i,j],permutation,instance))
			if profits[k] > max_profit:
				max_profit_permutation = copy.deepcopy(permutation)
				max_profit = profits[k]

		return [max_profit,max_profit_permutation]


def get_incremental_profit(selected_requests,instance):
	'''
	if incremental profit is less than or equal to zero then there is
	no way to match the selected requests in a way that satisfies SIR and is incremetally profitable
	'''

	result,result_permutation = get_profit(selected_requests,instance)
	for i in selected_requests:
		result -= get_profit([i],instance)[0] #TODO
	#TODO: Call get_profit()
	return [result,result_permutation]


#######################################################
#Two request case

instance = generate_instance()
all_requests = instance['all_requests'] #local pointer I think


#pprint(requests)
H = nx.Graph()
optimal_permutation_per_pair = {} #optimal in terms of incremental profit subject to SIR
for i in all_requests:
	for j in all_requests:
		if i != j:
			[incremental_profit,optimal_permutation] = get_incremental_profit([i,j],instance)	
			if  incremental_profit > 0:
				H.add_edge(str(i), str(j), weight=incremental_profit)
				optimal_permutation_per_pair[(i,j)] = optimal_permutation
			else:
				print "Pair {0},{1}  :   {2}".format(i,j,incremental_profit)


#nx.Draw(H)

#TODO: betas are not constants, but depend on detours. KEY