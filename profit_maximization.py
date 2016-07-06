import random, time, pulp, math, copy
import networkx as nx
import numpy as np
from pprint import pprint
from collections import OrderedDict
# import  matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
random.seed(5000)  # for replicability of experiments.
__author__ = 'q4fj4lj9'


def generate_instance():

	#requests and service provider details

	NO_OF_REQUESTS_IN_UNIVERSE = 100
	BETA1 = 0.9
	BETA2 = 0.8 #dummy for detour based discounting setting.

	GRID_MAX   = 100
	GRID_MAX_X = GRID_MAX
	GRID_MAX_Y = GRID_MAX

	MAX_DETOUR_SENSITIVITY = 100

	OUR_CUT_FROM_DRIVER = 0.3
	ALPHA_OP = MAX_DETOUR_SENSITIVITY/2

	GAMMA_ARRAY = [0.5*x for x in range(0,5)]

	PROB_PARAM_MARKET_SHARE = .6#300.0/ALPHA_OP

	PROB_PARAM_MARKET_SHARE_RIDE_SHARE_NO_GAMMA = 200

	PROB_PARAM_MARKET_SHARE_RIDE_SHARE_INTERNAL = 200

	PROB_PARAM_MARKET_SHARE_RIDE_SHARE_EXTERNAL = 0.5*PROB_PARAM_MARKET_SHARE_RIDE_SHARE_INTERNAL


	all_requests = OrderedDict()
	for i in range(NO_OF_REQUESTS_IN_UNIVERSE):
		all_requests[i] = OrderedDict()


		all_requests[i]['orig'] = (0,0)
		all_requests[i]['dest'] = (0,0)
		while all_requests[i]['orig'][0]==all_requests[i]['dest'][0] and all_requests[i]['orig'][1]==all_requests[i]['dest'][1]:
			all_requests[i]['orig'] = np.array([random.randint(0,GRID_MAX_X),random.randint(0,GRID_MAX_Y)])
			all_requests[i]['dest'] = np.array([random.randint(0,GRID_MAX_X),random.randint(0,GRID_MAX_Y)])

		all_requests[i]['detour_sensitivity'] = random.randint(1,MAX_DETOUR_SENSITIVITY)

		all_requests[i]['detour_sensitivity_normalized'] = all_requests[i]['detour_sensitivity']*1.0/ALPHA_OP

		all_requests[i]['our_cut_from_requester'] = {1:BETA1,2:BETA2}

		#some default values
		all_requests[i]['PROVIDER_MARKET'] = OrderedDict()
		all_requests[i]['PROVIDER_MARKET']['no_gamma'] = False
		for gamma in GAMMA_ARRAY:
			all_requests[i]['PROVIDER_MARKET'][gamma] = False
		
		#RIDE_SHARING key will be a dictionary with gamma as keys
		all_requests[i]['RIDE_SHARING'] = OrderedDict()
		all_requests[i]['RIDE_SHARING']['no_gamma'] = False
		for gamma in GAMMA_ARRAY:
			all_requests[i]['RIDE_SHARING'][gamma] = False
		

	#Splitting the universe into service provider part and non service provider part
	for i in all_requests:
		if random.uniform(0,1) < PROB_PARAM_MARKET_SHARE:
			all_requests[i]['PROVIDER_MARKET']['no_gamma'] = True
			for gamma in GAMMA_ARRAY:#redundant
				all_requests[i]['PROVIDER_MARKET'][gamma] = True


	#No SIR ridesharers in provider's market
	for i in all_requests:
		if all_requests[i]['PROVIDER_MARKET']['no_gamma'] == True:
			prob_threshold = PROB_PARAM_MARKET_SHARE_RIDE_SHARE_NO_GAMMA*\
				(1-all_requests[i]['our_cut_from_requester'][1])/ \
				all_requests[i]['detour_sensitivity']
			all_requests[i]['RIDE_SHARING']['no_gamma'] = (random.uniform(0,1) < prob_threshold)
			if all_requests[i]['RIDE_SHARING']['no_gamma'] == True:
				for gamma in GAMMA_ARRAY:#redundant
					all_requests[i]['RIDE_SHARING'][gamma] = True


	#Intoducing SIR as a function of gamma: Now, both internal (in PROVIDER_MARKET) and external (not in PROVIDER_MARKET) requests change their membership/preference


	previous_gamma = None
	for idx,current_gamma in enumerate(GAMMA_ARRAY):

		for i in all_requests:

			# print "idx: {0}, i = {1}".format(idx,i)

			prob_threshold = (1.0/(1+GAMMA_ARRAY[-1]))*\
				(1-all_requests[i]['our_cut_from_requester'][1])* \
				(1+current_gamma)/\
				all_requests[i]['detour_sensitivity']

			#Splitting the requests in the service provider part further into ride share and non-ride share
			if all_requests[i]['PROVIDER_MARKET']['no_gamma'] == True:

				prob_threshold *= PROB_PARAM_MARKET_SHARE_RIDE_SHARE_INTERNAL
	
				if idx == 0:
					if all_requests[i]['RIDE_SHARING']['no_gamma'] == False:
						all_requests[i]['RIDE_SHARING'][current_gamma] = (random.uniform(0,1) < prob_threshold)
					else:
						all_requests[i]['RIDE_SHARING'][current_gamma] = True
				elif all_requests[i]['RIDE_SHARING'][previous_gamma] == False: #flip a coin
					all_requests[i]['RIDE_SHARING'][current_gamma] = (random.uniform(0,1) < prob_threshold)
				elif all_requests[i]['RIDE_SHARING'][previous_gamma] == True: #copy previous val
					all_requests[i]['RIDE_SHARING'][current_gamma] = True


			#Adding requests not in original market share into the ride sharing pool
			if all_requests[i]['PROVIDER_MARKET']['no_gamma'] == False:

				prob_threshold *= PROB_PARAM_MARKET_SHARE_RIDE_SHARE_EXTERNAL

				if idx == 0:
					all_requests[i]['RIDE_SHARING'][current_gamma] = (random.uniform(0,1) < prob_threshold)
				elif all_requests[i]['RIDE_SHARING'][previous_gamma] == False:
					all_requests[i]['RIDE_SHARING'][current_gamma] = (random.uniform(0,1) < prob_threshold)
				elif all_requests[i]['RIDE_SHARING'][previous_gamma] == True:
					all_requests[i]['RIDE_SHARING'][current_gamma] = True
				
				if all_requests[i]['RIDE_SHARING'][current_gamma]==True:
					all_requests[i]['PROVIDER_MARKET'][current_gamma] = True
				
		previous_gamma = current_gamma

	#permutations needed for two participant matching
	all_permutations_two = {('si','sj','di','dj'):'case1',
			('si','sj','dj','di'):'case2',
			('sj','si','dj','di'):'case1',
			('sj','si','di','dj'):'case2'}#TODO: make more comprehensible

	#redundant array
	GAMMA_ARRAY_ALL = ['no_gamma']
	GAMMA_ARRAY_ALL.extend(GAMMA_ARRAY)

	return {'all_requests': all_requests,
	'instance_params':{
	'OUR_CUT_FROM_DRIVER': OUR_CUT_FROM_DRIVER,
	'ALPHA_OP':ALPHA_OP,
	'all_permutations_two': all_permutations_two,
	'GAMMA_ARRAY':GAMMA_ARRAY,
	'GAMMA_ARRAY_ALL': GAMMA_ARRAY_ALL}}

def euclidean(x,y):
	assert x is not None and y is not None
	return np.linalg.norm(x - y)

def get_driving_distance(focus_request,selected_requests,permutation,instance):

	#local
	all_requests = instance['all_requests']
	all_permutations_two = instance['instance_params']['all_permutations_two']

	#Logic
	if len(selected_requests) > 2:
		return NotImplementedError

	if len(selected_requests) == 1: #redundant
		return euclidean(all_requests[selected_requests[0]]['orig'],all_requests[selected_requests[0]]['dest'])

	if len(selected_requests) == 2:

		i = selected_requests[0]
		j = selected_requests[1]

		#None case remains the same
		if focus_request == i:
			focus_request = 'i'
		elif focus_request == j:
			focus_request = 'j'

		if permutation[0]=='sj':#i and j are symmetric within each case. See board photo. TODO

			#None case remains the same
			if focus_request == 'i':
				focus_request = 'j'
			elif focus_request == 'j':
				focus_request = 'i'
			
			(i,j) = (j,i) #swap

	
		if all_permutations_two[permutation] == 'case1':
			
			if focus_request == None:
				return euclidean(all_requests[i]['orig'],all_requests[j]['orig']) + \
						euclidean(all_requests[j]['orig'],all_requests[i]['dest']) + \
						euclidean(all_requests[i]['dest'],all_requests[j]['dest'])

			elif focus_request == 'i':
				return euclidean(all_requests[i]['orig'],all_requests[j]['orig']) + \
						euclidean(all_requests[j]['orig'],all_requests[i]['dest'])

			elif focus_request == 'j':
				return euclidean(all_requests[j]['orig'],all_requests[i]['dest']) + \
						euclidean(all_requests[i]['dest'],all_requests[j]['dest'])
			else:
				raise Exception


		elif all_permutations_two[permutation] == 'case2':

			if focus_request == None or focus_request == 'i':

				return euclidean(all_requests[i]['orig'],all_requests[j]['orig']) + \
						euclidean(all_requests[j]['orig'],all_requests[j]['dest']) + \
						euclidean(all_requests[j]['dest'],all_requests[i]['dest'])
			elif focus_request == 'j':
				return euclidean(all_requests[j]['orig'],all_requests[j]['dest'])
			else:
				raise Exception
		else:
			raise Exception
	else:
		raise Exception

def check_SIR(selected_requests,permutation,instance,experiment_params):
	
	#if SIR constraints are not applicable, then we should not check SIR
	if experiment_params['GAMMA']=='no_gamma':
		return True
	elif experiment_params['DISCOUNT_SETTING'] == 'detour_based':
		return check_SIR_satisfaction_detour_based(selected_requests,permutation,instance,experiment_params)
	elif experiment_params['DISCOUNT_SETTING'] == 'independent':
		return check_SIR_satisfaction_general(selected_requests,permutation,instance,experiment_params)
	else:
		return NotImplementedError

def check_SIR_satisfaction_general(selected_requests,permutation,instance,experiment_params):
	
	all_requests = instance['all_requests'] #local copy
	all_permutations_two = instance['instance_params']['all_permutations_two']

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

		if all_permutations_two[permutation] == 'case1':

			constraint_i =  (all_requests[i]['our_cut_from_requester'][1] - all_requests[i]['our_cut_from_requester'][2])*euclidean(all_requests[i]['orig'],all_requests[i]['dest']) \
				>= (experiment_params['GAMMA'] + all_requests[i]['detour_sensitivity_normalized'])*\
					(euclidean(all_requests[i]['orig'],all_requests[j]['orig']) + \
					euclidean(all_requests[j]['orig'],all_requests[i]['dest']) - \
					euclidean(all_requests[i]['orig'],all_requests[i]['dest']))
			constraint_j = (all_requests[j]['our_cut_from_requester'][1] - all_requests[j]['our_cut_from_requester'][2])*euclidean(all_requests[j]['orig'],all_requests[j]['dest']) \
				>= (experiment_params['GAMMA'] + all_requests[j]['detour_sensitivity_normalized'])*\
					(euclidean(all_requests[j]['orig'],all_requests[i]['dest']) + \
					euclidean(all_requests[i]['dest'],all_requests[j]['dest']) - \
					euclidean(all_requests[j]['orig'],all_requests[j]['dest']))

		elif all_permutations_two[permutation] == 'case2':

			constraint_i = (all_requests[i]['our_cut_from_requester'][1] - all_requests[i]['our_cut_from_requester'][2])*euclidean(all_requests[i]['orig'],all_requests[i]['dest']) \
				>= (experiment_params['GAMMA'] + all_requests[i]['detour_sensitivity_normalized'])*\
					(euclidean(all_requests[i]['orig'],all_requests[j]['orig']) + \
					euclidean(all_requests[j]['orig'],all_requests[j]['dest']) + \
					euclidean(all_requests[j]['dest'],all_requests[i]['dest']) - \
					euclidean(all_requests[i]['orig'],all_requests[i]['dest']))
			constraint_j = all_requests[j]['our_cut_from_requester'][1] >= all_requests[j]['our_cut_from_requester'][2]

		else:
			print "Error!" #TODO
			constraint_i = False
			constraint_j = False
			return NotImplementedError #todo: change to appropriate error

		return constraint_i and constraint_j

def check_SIR_satisfaction_detour_based(selected_requests,permutation,instance,experiment_params):
	
	all_requests = instance['all_requests'] #local copy
	all_permutations_two = instance['instance_params']['all_permutations_two']


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
				>= experiment_params['GAMMA'] + all_requests[i]['detour_sensitivity_normalized']
		constraint_j = all_requests[j]['our_cut_from_requester'][1]\
				>= experiment_params['GAMMA'] + all_requests[j]['detour_sensitivity_normalized']

		if all_permutations_two[permutation] == 'case1':

			return constraint_i and constraint_j

		elif all_permutations_two[permutation] == 'case2':

			return constraint_i

		else:
			print "Error!" #TODO
			return False

def get_profit_non_ride_sharing(selected_request_as_list,instance):
	assert instance is not None
	assert selected_request_as_list is not None	
	#single id is input as a list to maintain API consistency with get_profit() function

	#print 'selected_request_as_list',selected_request_as_list

	return instance['instance_params']['ALPHA_OP']*\
		instance['instance_params']['OUR_CUT_FROM_DRIVER']*\
				euclidean(instance['all_requests'][selected_request_as_list[0]]['orig'],instance['all_requests'][selected_request_as_list[0]]['dest'])

def get_profit_unmatched(selected_request_as_list,instance):
	assert instance is not None
	assert selected_request_as_list is not None	
	#single id is input as a list to maintain API consistency with get_profit() function

	return instance['instance_params']['ALPHA_OP']*\
				(instance['all_requests'][selected_request_as_list[0]]['our_cut_from_requester'][1] - \
					(1 - instance['instance_params']['OUR_CUT_FROM_DRIVER']))*\
				euclidean(instance['all_requests'][selected_request_as_list[0]]['orig'],instance['all_requests'][selected_request_as_list[0]]['dest'])

def get_profit_matched(selected_requests,instance,permutation,experiment_params):
	
	assert len(selected_requests) == 2

	instance_params = instance['instance_params']
	all_requests = instance['all_requests']


	if experiment_params['DISCOUNT_SETTING']=='independent':

		return instance_params['ALPHA_OP']*\
				(sum([all_requests[m]['our_cut_from_requester'][2]*\
					euclidean(all_requests[m]['orig'],all_requests[m]['dest']) for m in selected_requests]) - \
				(1-instance_params['OUR_CUT_FROM_DRIVER'])*get_driving_distance(None,selected_requests,permutation,instance))
	else:

		result = 0
		for m in selected_requests:
			beta2 = all_requests[m]['our_cut_from_requester'][1]
			beta2 *= (1 - (get_driving_distance(m,selected_requests,permutation,instance) - \
				euclidean(all_requests[m]['orig'],all_requests[m]['dest']))*\
				1.0/euclidean(all_requests[m]['orig'],all_requests[m]['dest']))

			result +=beta2*euclidean(all_requests[m]['orig'],all_requests[m]['dest'])

		return instance_params['ALPHA_OP']*result - \
			(1-instance_params['OUR_CUT_FROM_DRIVER'])*get_driving_distance(None,selected_requests,permutation,instance)

def get_incremental_profit(selected_requests,instance,experiment_params):

	'''
	if incremental profit is less than or equal to zero then there is
	no way to match the selected requests in a way that satisfies SIR and is incremetally profitable
	'''

	assert instance is not None
	assert selected_requests is not None

	#Local copies
	all_requests = instance['all_requests']
	instance_params = instance['instance_params']
	all_permutations_two = instance['instance_params']['all_permutations_two']

	#Logic
	if len(selected_requests) > 2:
		return NotImplementedError

	if len(selected_requests) == 1:
		#ideally this should not be executed, since incremental profit is zero here.
		max_profit =  get_profit_unmatched(selected_requests,instance)
		max_profit_permutation =  ('si','di')

	if len(selected_requests) == 2:
		i = selected_requests[0]
		j = selected_requests[1]

		profits = np.zeros(len(all_permutations_two))
		max_profit = 0
		max_profit_permutation = None
		for k,permutation in enumerate(all_permutations_two):
			SIR_satisfied = check_SIR([i,j],permutation,instance, experiment_params)
			if SIR_satisfied is True:
				profits[k] = get_profit_matched([i,j],instance,permutation,experiment_params)
			if profits[k] > max_profit:
				max_profit_permutation = copy.deepcopy(permutation)
				max_profit = profits[k]

	#assumes max_profit exists
	#to get incremental profit, subtracting the profits from their dedicated rides, have to give discounts because they wanted to rideshare but we couldn't match them
	for i in selected_requests:
		max_profit -= get_profit_unmatched([i],instance)
	return [max_profit,max_profit_permutation]

def is_interested_in_ridesharing(i,instance,experiment_params):
	GAMMA = experiment_params['GAMMA']
	if instance['all_requests'][i]['PROVIDER_MARKET'][GAMMA]==True:#maybe redundant
		if instance['all_requests'][i]['RIDE_SHARING'][GAMMA]==True:
			return True
	return False #default

def match_requests(instance,experiment_params):

	H = nx.Graph()
	request_pairs_with_permutations = {}
	non_ridesharing_requests = []
	unmatched_requests_initial = set()
	for i in instance['all_requests']:
		if is_interested_in_ridesharing(i,instance,experiment_params) is False:
			# print "{0} is not ridesharing.".format(i)
			non_ridesharing_requests.append(i)
			continue
		for j in instance['all_requests']:
			if i < j and is_interested_in_ridesharing(j,instance,experiment_params) is True:#cover the symmetric relation as well the the case i!=j

				#adding both as candidates that wanted to rideshare but could not be matched
				unmatched_requests_initial.add(i)
				unmatched_requests_initial.add(j)
				#print "i,j = {0},{1}".format(i,j)

				#get a permutation and the incremental profit for the pair i,j
				[incremental_profit,optimal_permutation] = get_incremental_profit([i,j],instance,experiment_params)

				if  incremental_profit > 0:
					H.add_edge(str(i), str(j), weight=incremental_profit)
					request_pairs_with_permutations[(i,j)] = optimal_permutation
					# print "Include edge {0},{1}  :   {2}".format(i,j,incremental_profit)

	#these requests failed to satisfy incremental profit > 0 for any of their potential ridesharing neighbors
	#pprint(unmatched_requests_initial)
	#print H.nodes()
	unmatched_requests_initial = [x for x in unmatched_requests_initial if str(x) not in H.nodes()]
	#pprint(unmatched_requests_initial)

	print "Size of matching graph:",len(H.nodes())

	mates = nx.max_weight_matching(H)
	matched_request_pairs_with_permutations = {}
	for i,j in mates.items():
		if int(i) < int(j):
			# print "types: {0},{1},{2},{3}".format(i,type(i),j,type(j))
			matched_request_pairs_with_permutations[(int(i),int(j))] = request_pairs_with_permutations[(int(i),int(j))]
	unmatched_requests = [int(x) for x in H.nodes() if x not in mates.keys()]
	#pprint(unmatched_requests)
	if len(unmatched_requests_initial)>0:
		if len(unmatched_requests) >0:
			unmatched_requests.extend(unmatched_requests_initial)
		else:
			unmatched_requests = unmatched_requests_initial

	result =  {'non_ridesharing_requests':non_ridesharing_requests,
	'unmatched_requests':unmatched_requests,
	'matched_request_pairs_with_permutations':matched_request_pairs_with_permutations}

	# pprint(result)

	return result

def get_profit_from_unmatched_requests(selected_requests,instance):

	if selected_requests is None:
		return 0
	elif len(selected_requests) == 0:
		return 0

	result = 0
	for i in selected_requests:
		result += get_profit_unmatched([i],instance)

	return result

def get_profit_from_non_ridesharing_requests(selected_requests,instance):

	if selected_requests is None:
		return 0
	elif len(selected_requests) == 0:
		return 0


	#print 'selected_requests',selected_requests

	result = 0
	for i in selected_requests:
		result += get_profit_non_ride_sharing([i],instance)

	return result

def get_profit_from_matched_requests(matched_request_pairs_with_permutations,instance,experiment_params):

	if len(matched_request_pairs_with_permutations) == 0:
		return 0

	result = 0
	for request_pairs in matched_request_pairs_with_permutations:
		result += get_profit_matched(request_pairs,instance,matched_request_pairs_with_permutations[request_pairs],experiment_params)
	return result

def solve_instance(instance,experiment_params):

	solution = match_requests(instance,experiment_params)
	total_profit = \
		get_profit_from_non_ridesharing_requests(solution['non_ridesharing_requests'],instance) + \
		get_profit_from_unmatched_requests(solution['unmatched_requests'],instance) + \
		get_profit_from_matched_requests(solution['matched_request_pairs_with_permutations'],instance,experiment_params)

	return solution,total_profit

def get_stats(instance):
	#simple helper function to display
	all_gammas = ['no_gamma']
	all_gammas.extend(instance['instance_params']['GAMMA_ARRAY'])

	result = []
	for i in instance['all_requests']:
		if instance['all_requests'][i]['PROVIDER_MARKET']['no_gamma']==True:
			result.append(i)
			
	print "\nRequests in provider market share initially w/o SIR-Gamma:",len(result)
	print(result)

	result2 = [x for x in instance['all_requests'] if x not in result]
	print "\nRequests NOT in provider market share initially w/o SIR-Gamma:",len(result2)
	print(result2)

	print "\nRequests in provider market initially:"
	result = []
	for gamma in all_gammas:
		for i in instance['all_requests']:
			if i in result:
				continue
			if instance['all_requests'][i]['PROVIDER_MARKET']['no_gamma']==True:
				if instance['all_requests'][i]['RIDE_SHARING'][gamma]==True:
					print "{0} in ridesharing at gamma = {1}".format(i,gamma)
					result.append(i)

	
	print "\nOutside provider market:"
	result = []
	for gamma in all_gammas:
		for i in instance['all_requests']:
			if i in result:
				continue
			if instance['all_requests'][i]['PROVIDER_MARKET']['no_gamma']==False:
				if instance['all_requests'][i]['RIDE_SHARING'][gamma]==True:
					print "{0} in ridesharing at gamma = {1}".format(i,gamma)
					result.append(i)


	for gamma in instance['instance_params']['GAMMA_ARRAY_ALL']:
		print "No of potential ridesharers at Gamma = {0}: {1}".format(gamma,len([x for x in instance['all_requests'] if instance['all_requests'][x]['RIDE_SHARING'][gamma]==True]))

	for gamma in instance['instance_params']['GAMMA_ARRAY_ALL']:
		print "No of potential ridesharers at Gamma = {0}: {1}".format(gamma,len([x for x in instance['all_requests'] if instance['all_requests'][x]['RIDE_SHARING'][gamma]==True and instance['all_requests'][x]['PROVIDER_MARKET'][gamma]==True]))

	data = np.zeros((len(instance['all_requests']),len(instance['instance_params']['GAMMA_ARRAY_ALL'])))
	for x in instance['all_requests']:
		request = instance['all_requests'][x]
		# for i,e in enumerate(request['PROVIDER_MARKET']):
		# 	if request['PROVIDER_MARKET'][e] is True:
		# 		data[x,i] = 1
		# 	else:
		# 		data[x,i] = 0
		for i,e in enumerate(request['RIDE_SHARING']):
			if request['RIDE_SHARING'][e] is True:
				data[x,i] = 1
			else:
				data[x,i] = 0
	# for i in range(data.shape[0]):
		# print i,data[i]

def get_experiment_params(instance,GAMMA='no_gamma'):

	DISCOUNT_SETTING = 'detour_based' # 'independent'
	assert GAMMA in instance['instance_params']['GAMMA_ARRAY_ALL']

	return {'DISCOUNT_SETTING':DISCOUNT_SETTING,
	'GAMMA':GAMMA}

if __name__=='__main__':

	instance = generate_instance()
	get_stats(instance)

	do_experiment = True
	if do_experiment is True:
		for gamma in instance['instance_params']['GAMMA_ARRAY_ALL']:
			print 'Gamma = ',gamma
			experiment_params = get_experiment_params(instance,GAMMA=gamma)
			solution,total_profit = solve_instance(instance,experiment_params)
			print "total profit:",total_profit
			# pprint(solution)