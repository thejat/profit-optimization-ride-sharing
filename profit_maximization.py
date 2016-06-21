import random, time, pulp, matplotlib, math, copy
import networkx as nx
import numpy as np
from pprint import pprint
from collections import OrderedDict
matplotlib.use('Agg')
import matplotlib.pyplot as plt
random.seed(1000)  # for replicability of experiments.
__author__ = 'q4fj4lj9'




def generate_instance():

	#requests and service provider details

	NO_OF_REQUESTS_IN_UNIVERSE = 20
	INITIAL_MARKET_SHARE_OF_SERVICE_PROVIDER = .6
	INITIAL_RIDE_SHARING_AMONG_SERVICE_PROVIDER = .3
	BETA1 = 0.9
	BETA2 = 0.8 #dummy for detour based discounting setting.

	GRID_MAX   = 100
	GRID_MAX_X = GRID_MAX
	GRID_MAX_Y = GRID_MAX

	MAX_SENSITIVITY = 1000

	OUR_CUT_FROM_DRIVER = 0.3
	ALPHA_OP = MAX_SENSITIVITY/2

	all_requests = OrderedDict()
	for i in range(NO_OF_REQUESTS_IN_UNIVERSE):
		all_requests[i] = OrderedDict()


		all_requests[i]['orig'] = (0,0)
		all_requests[i]['dest'] = (0,0)
		while all_requests[i]['orig'][0]==all_requests[i]['dest'][0] and all_requests[i]['orig'][1]==all_requests[i]['dest'][1]:
			all_requests[i]['orig'] = np.array([random.randint(0,GRID_MAX_X),random.randint(0,GRID_MAX_Y)])
			all_requests[i]['dest'] = np.array([random.randint(0,GRID_MAX_X),random.randint(0,GRID_MAX_Y)])

		all_requests[i]['detour_sensitivity'] = random.randint(1,MAX_SENSITIVITY)

		all_requests[i]['detour_sensitivity_normalized'] = all_requests[i]['detour_sensitivity']*1.0/ALPHA_OP

		all_requests[i]['our_cut_from_requester'] = {1:BETA1,2:BETA2}

	#Splitting the universe into service provider part and non service provider part

	request_ids_initial_market_share = random.sample(all_requests.keys(),int(INITIAL_MARKET_SHARE_OF_SERVICE_PROVIDER*NO_OF_REQUESTS_IN_UNIVERSE))


	for i in all_requests:
		if i in request_ids_initial_market_share:
			all_requests[i]['PROVIDER_MARKET'] = True
		else:
			all_requests[i]['PROVIDER_MARKET'] = False


	#Splitting the requests in the service provider part further into ride share and non-ride share
	request_ids_initial_ride_sharing = random.sample(request_ids_initial_market_share,int(INITIAL_RIDE_SHARING_AMONG_SERVICE_PROVIDER*len(request_ids_initial_market_share)))

	for i in request_ids_initial_market_share:
		if i in request_ids_initial_ride_sharing:
			all_requests[i]['RIDE_SHARING'] = True
		else:
			all_requests[i]['RIDE_SHARING'] = False

	return {'all_requests': all_requests,
	'params':{
	'OUR_CUT_FROM_DRIVER': OUR_CUT_FROM_DRIVER,
	'ALPHA_OP':ALPHA_OP}}

def euclidean(x,y):
	assert x is not None and y is not None
	return np.linalg.norm(x - y)

def get_driving_distance(selected_requests,permutation,all_permutations,instance):

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

def check_SIR_satisfaction_general(selected_requests,permutation,instance,all_permutations):
	
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

			constraint_i =  (all_requests[i]['our_cut_from_requester'][1] - all_requests[i]['our_cut_from_requester'][2])*euclidean(all_requests[i]['orig'],all_requests[i]['dest']) \
				>= (params['GAMMA'] + all_requests[i]['detour_sensitivity_normalized'])*\
					(euclidean(all_requests[i]['orig'],all_requests[j]['orig']) + \
					euclidean(all_requests[j]['orig'],all_requests[i]['dest']) - \
					euclidean(all_requests[i]['orig'],all_requests[i]['dest']))
			constraint_j = (all_requests[j]['our_cut_from_requester'][1] - all_requests[j]['our_cut_from_requester'][2])*euclidean(all_requests[j]['orig'],all_requests[j]['dest']) \
				>= (params['GAMMA'] + all_requests[j]['detour_sensitivity_normalized'])*\
					(euclidean(all_requests[j]['orig'],all_requests[i]['dest']) + \
					euclidean(all_requests[i]['dest'],all_requests[j]['dest']) - \
					euclidean(all_requests[j]['orig'],all_requests[j]['dest']))

		elif all_permutations[permutation] == 'case2':

			constraint_i = (all_requests[i]['our_cut_from_requester'][1] - all_requests[i]['our_cut_from_requester'][2])*euclidean(all_requests[i]['orig'],all_requests[i]['dest']) \
				>= (params['GAMMA'] + all_requests[i]['detour_sensitivity_normalized'])*\
					(euclidean(all_requests[i]['orig'],all_requests[j]['orig']) + \
					euclidean(all_requests[j]['orig'],all_requests[j]['dest']) + \
					euclidean(all_requests[j]['dest'],all_requests[i]['dest']) - \
					euclidean(all_requests[i]['orig'],all_requests[i]['dest']))
			constraint_j = all_requests[j]['our_cut_from_requester'][1] >= all_requests[j]['our_cut_from_requester'][2]

		else:
			print "Error!" #TODO
			constraint_i = False
			constraint_j = False

		return constraint_i and constraint_j

def check_SIR_satisfaction_detour_based(selected_requests,permutation,instance,all_permutations):
	
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

		constraint_i = all_requests[i]['our_cut_from_requester'][1]\
				>= params['GAMMA'] + all_requests[i]['detour_sensitivity_normalized']
		constraint_j = all_requests[j]['our_cut_from_requester'][1]\
				>= params['GAMMA'] + all_requests[j]['detour_sensitivity_normalized']

		if all_permutations[permutation] == 'case1':

			return constraint_i and constraint_j

		elif all_permutations[permutation] == 'case2':

			return constraint_i

		else:
			print "Error!" #TODO
			return False

def get_profit(selected_requests,instance):
	assert instance is not None
	assert selected_requests is not None

	#Local copies
	all_requests = instance['all_requests']
	params = instance['params']

	if params['EXP_DISCOUNT_SETTING'] == 'detour_based':
		check_SIR_routine = check_SIR_satisfaction_detour_based
	elif params['EXP_DISCOUNT_SETTING'] == 'independent':
		check_SIR_routine = check_SIR_satisfaction_general

	#Logic
	if len(selected_requests) > 2:
		return NotImplementedError

	if len(selected_requests) == 1:
		return [all_requests[selected_requests[0]]['our_cut_from_requester'][1]*euclidean(all_requests[selected_requests[0]]['orig'],all_requests[selected_requests[0]]['dest']),('si','di')]

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
			SIR_satisfied = check_SIR_routine([i,j],permutation,instance, all_permutations)
			if SIR_satisfied is True:
				profits[k] = params['ALPHA_OP']*(sum([all_requests[m]['our_cut_from_requester'][2]*euclidean(all_requests[m]['orig'],all_requests[m]['dest']) for m in (i,j)]) - (1-params['OUR_CUT_FROM_DRIVER'])*get_driving_distance([i,j],permutation,all_permutations,instance))
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

def get_experiment_params():

	DISCOUNT_SETTING = 'detour_based' # 'independent'
	SIR_SETTING = False
	GAMMA = 0 #valid only when SIR_SETTING is True

	return {'DISCOUNT_SETTING':DISCOUNT_SETTING,
	'SIR_SETTING':SIR_SETTING,
	'GAMMA':GAMMA}


def is_ridesharing(i,instance):
	if instance['all_requests'][i]['PROVIDER_MARKET']==True:
		if instance['all_requests'][i]['RIDE_SHARING']==True:
			return True
	return False

def match_requests(instance,experiment_params):

	H = nx.Graph()
	request_pairs_with_permutations = {}
	non_ridesharing_requests = []
	for i in instance['all_requests']:
		if is_ridesharing(i,instance) is False:
			non_ridesharing_requests.append(i)
			continue
		for j in instance['all_requests']:
			if i != j and is_ridesharing(j,instance) is True:
				[incremental_profit,optimal_permutation] = get_incremental_profit([i,j],instance,experiment_params)	#TODO CHANGE FUNCTION
				if  incremental_profit > 0:
					H.add_edge(str(i), str(j), weight=incremental_profit)
					request_pairs_with_permutations[(i,j)] = optimal_permutation
					print "Include edge {0},{1}  :   {2}".format(i,j,incremental_profit)


	mates = nx.max_weight_matching(H)
	matched_request_pairs_with_permutations = {}
	for i,j in mates.items():
		if i < j:
			matched_request_pairs_with_permutations[(i,j)] = request_pairs_with_permutations[(i,j)]
	unmatched_requests = [x in H.nodes() if x not in mates.keys()]

	return {'non_ridesharing_requests':non_ridesharing_requests,
	'unmatched_requests':unmatched_requests,
	'matched_request_pairs_with_permutations':matched_request_pairs_with_permutations}


def get_profit_from_unmatched_requests(requests,instance):
	return NotImplementedError

def get_profit_from_non_ridesharing_requests(requests,instance):
	return NotImplementedError

def get_profit_from_matched_requests(matched_request_pairs_with_permutations,instance):
	return NotImplementedError

if '__init__'==__main__:

	instance = generate_instance()
	experiment_params = get_experiment_params()

	[non_ridesharing_requests,unmatched_requests,matched_request_pairs_with_permutations] = match_requests(instance,experiment_params)

	total_profit = \
		get_profit_from_non_ridesharing_requests(non_ridesharing_requests,instance) + \
		get_profit_from_unmatched_requests(unmatched_requests,instance) + \
		get_profit_from_matched_requests(matched_request_pairs_with_permutations,instance)

	print total_profit