import random, time, math, copy, numpy, pickle
import networkx as nx
import pandas
from pprint import pprint
from collections import OrderedDict
# import  matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
random.seed(5000)  # for replicability of experiments.
__author__ = 'q4fj4lj9'


def load_nyc_data():
	df0 = pandas.read_csv('../../../Xharecost_MS_annex/hour9.csv',usecols= ['pickup_datetime','passenger_count','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude'])

	nyc_df = df0[(abs(df0.pickup_latitude-0) > 1e-5) & \
         (abs(df0.pickup_longitude-0) > 1e-5) & \
         (abs(df0.dropoff_latitude-0) > 1e-5) & \
         (abs(df0.dropoff_longitude-0) > 1e-5)]
	nyc_df['pickup_datetime'] = pandas.to_datetime(nyc_df['pickup_datetime'])
	return nyc_df

def get_nyc_data(nyc_df,instance_no=0):
	
	assert instance_no >=0 and instance_no <= 60
	assert type(instance_no) is int

	data = nyc_df[(nyc_df['pickup_datetime'].dt.minute >= instance_no) & \
					(nyc_df['pickup_datetime'].dt.minute < instance_no+1)]
	print 'NYC instance for minute: {0} count of requests: {1}'.format(instance_no,len(data))
	return data

def get_instance_params(NO_OF_REQUESTS_IN_UNIVERSE=100,GAMMA_ARRAY=[0,0.1,0.2,0.3,0.4],flag_nyc_data=False):

	#requests and service provider details
	BETA1 = 0.9
	OUR_CUT_FROM_REQUESTER_COMMON = {1:BETA1,2:0.9*BETA1} #second key:value pair is irrelevant for detour based discount
	MAX_DETOUR_SENSITIVITY = 100

	OUR_CUT_FROM_DRIVER = 0.3
	ALPHA_OP = MAX_DETOUR_SENSITIVITY/2

	#permutations needed for two participant matching
	all_permutations_two = {('si','sj','di','dj'):'case1',
			('si','sj','dj','di'):'case2',
			('sj','si','dj','di'):'case1',
			('sj','si','di','dj'):'case2'}#TODO: make more comprehensible

	# GAMMA_ARRAY = [float("%.3f" % (BETA1**(2*x+1))) for x in range(10,-1,-2)]
	# GAMMA_ARRAY.append(0)
	GAMMA_ARRAY = sorted(GAMMA_ARRAY) #see default in function arg definition
	GAMMA_ARRAY_ALL = ['no_gamma'] 	#redundant array
	GAMMA_ARRAY_ALL.extend(GAMMA_ARRAY)

	return {
	'NO_OF_REQUESTS_IN_UNIVERSE':NO_OF_REQUESTS_IN_UNIVERSE,
	'OUR_CUT_FROM_REQUESTER_COMMON':OUR_CUT_FROM_REQUESTER_COMMON,
	'MAX_DETOUR_SENSITIVITY': MAX_DETOUR_SENSITIVITY,
	'OUR_CUT_FROM_DRIVER': OUR_CUT_FROM_DRIVER,
	'ALPHA_OP':ALPHA_OP,
	'all_permutations_two': all_permutations_two,
	'GAMMA_ARRAY':GAMMA_ARRAY,
	'GAMMA_ARRAY_ALL': GAMMA_ARRAY_ALL,
	'flag_nyc_data':flag_nyc_data}

def generate_base_instance(instance_params,flag_nyc_data=False,instance_no=0,nyc_df=None):

	if instance_params['flag_nyc_data']==False:
		GRID_MAX   = 100
		GRID_MAX_X = GRID_MAX
		GRID_MAX_Y = GRID_MAX
	else:
		odPatterns = get_nyc_data(nyc_df,instance_no)
		instance_params['NO_OF_REQUESTS_IN_UNIVERSE'] = len(odPatterns)

	# GENERATE OD locations for all requests
	all_requests = OrderedDict()
	for i in range(instance_params['NO_OF_REQUESTS_IN_UNIVERSE']):
		all_requests[i] = OrderedDict()

		if instance_params['flag_nyc_data']==False:
			all_requests[i]['orig'] = (0,0)
			all_requests[i]['dest'] = (0,0)
			while all_requests[i]['orig'][0]==all_requests[i]['dest'][0] and all_requests[i]['orig'][1]==all_requests[i]['dest'][1]:
				all_requests[i]['orig'] = numpy.array([random.randint(0,GRID_MAX_X),random.randint(0,GRID_MAX_Y)])
				all_requests[i]['dest'] = numpy.array([random.randint(0,GRID_MAX_X),random.randint(0,GRID_MAX_Y)])
		else:
			all_requests[i]['orig'] = (float(odPatterns.iloc[[i]]['pickup_longitude']),float(odPatterns.iloc[[i]]['pickup_latitude']))
			all_requests[i]['dest'] = (float(odPatterns.iloc[[i]]['dropoff_longitude']),float(odPatterns.iloc[[i]]['dropoff_latitude']))
			# print all_requests[i]['orig'],all_requests[i]['dest']

		all_requests[i]['detour_sensitivity'] = random.randint(1,instance_params['MAX_DETOUR_SENSITIVITY'])

		all_requests[i]['detour_sensitivity_normalized'] = all_requests[i]['detour_sensitivity']*1.0/instance_params['ALPHA_OP']

		all_requests[i]['our_cut_from_requester'] = instance_params['OUR_CUT_FROM_REQUESTER_COMMON']

		#some default values
		all_requests[i]['PROVIDER_MARKET'] = OrderedDict()
		all_requests[i]['PROVIDER_MARKET']['no_gamma'] = False
		for gamma in instance_params['GAMMA_ARRAY']:
			all_requests[i]['PROVIDER_MARKET'][gamma] = False
		
		#RIDE_SHARING key will be a dictionary with gamma as keys
		all_requests[i]['RIDE_SHARING'] = OrderedDict()
		all_requests[i]['RIDE_SHARING']['no_gamma'] = False
		for gamma in instance_params['GAMMA_ARRAY']:
			all_requests[i]['RIDE_SHARING'][gamma] = False

		#For Logging purposes: RIDE_SHARING_BIAS key will be a dictionary with gamma as keys
		all_requests[i]['RIDE_SHARING_BIAS'] = OrderedDict()
		all_requests[i]['RIDE_SHARING_BIAS']['no_gamma'] = -1 #Convention: NO FLIP means negative value
		for gamma in instance_params['GAMMA_ARRAY']:
			all_requests[i]['RIDE_SHARING_BIAS'][gamma] = -1 #Convention: NO FLIP means negative value
		
	#Flipping coins is moved to its own function

	return {'all_requests': all_requests,
	'instance_params':instance_params}

def flip_coins_wo_gamma(instance_base,coin_flip_params):

	instance_partial = copy.deepcopy(instance_base)

	#local copy
	all_requests = instance_partial['all_requests']
	GAMMA_ARRAY  = instance_partial['instance_params']['GAMMA_ARRAY']
	PROB_PARAM_MARKET_SHARE 					= coin_flip_params['PROB_PARAM_MARKET_SHARE']
	PROB_PARAM_MARKET_SHARE_RIDE_SHARE_NO_GAMMA = coin_flip_params['PROB_PARAM_MARKET_SHARE_RIDE_SHARE_NO_GAMMA']

	#COIN FLIPS: Splitting the universe into service provider part and non service provider part
	for i in all_requests:
		if random.uniform(0,1) < PROB_PARAM_MARKET_SHARE:
			all_requests[i]['PROVIDER_MARKET']['no_gamma'] = True
			for gamma in GAMMA_ARRAY:#redundant
				all_requests[i]['PROVIDER_MARKET'][gamma] = True

	#COIN FLIPS: No SIR ridesharers in provider's market
	for i in all_requests:
		if all_requests[i]['PROVIDER_MARKET']['no_gamma'] == True:
			prob_threshold = PROB_PARAM_MARKET_SHARE_RIDE_SHARE_NO_GAMMA*\
				(1-all_requests[i]['our_cut_from_requester'][1])/ \
				all_requests[i]['detour_sensitivity']
			all_requests[i]['RIDE_SHARING']['no_gamma'] = (random.uniform(0,1) < prob_threshold)
			if all_requests[i]['RIDE_SHARING']['no_gamma'] == True:
				for gamma in GAMMA_ARRAY:#redundant
					all_requests[i]['RIDE_SHARING'][gamma] = True

			all_requests[i]['RIDE_SHARING_BIAS']['no_gamma'] = prob_threshold



	instance_partial['instance_params']['PROB_PARAM_MARKET_SHARE'] 	= coin_flip_params['PROB_PARAM_MARKET_SHARE']
	instance_partial['instance_params']['PROB_PARAM_MARKET_SHARE_RIDE_SHARE_NO_GAMMA'] = coin_flip_params['PROB_PARAM_MARKET_SHARE_RIDE_SHARE_NO_GAMMA']

	instance_partial['all_requests'] = all_requests
	return instance_partial

def flip_coins_w_gamma(instance_partial,coin_flip_params):

	instance = copy.deepcopy(instance_partial)

	#local copy
	all_requests = instance['all_requests']
	GAMMA_ARRAY  = instance['instance_params']['GAMMA_ARRAY']
	PROB_PARAM_MARKET_SHARE_RIDE_SHARE_INTERNAL = coin_flip_params['PROB_PARAM_MARKET_SHARE_RIDE_SHARE_INTERNAL']
	PROB_PARAM_MARKET_SHARE_RIDE_SHARE_EXTERNAL = coin_flip_params['PROB_PARAM_MARKET_SHARE_RIDE_SHARE_EXTERNAL']
	gamma_offset = coin_flip_params['GAMMA_OFFSET']

	#COIN FLIPS: Intoducing SIR as a function of gamma: Now, both internal (in PROVIDER_MARKET) and external (not in PROVIDER_MARKET) requests change their membership/preference
	previous_gamma = None
	for idx,current_gamma in enumerate(GAMMA_ARRAY):

		for i in all_requests:

			# print "idx: {0}, i = {1}".format(idx,i)

			prob_threshold = (1.0/(1+GAMMA_ARRAY[-1]))*\
				(1-all_requests[i]['our_cut_from_requester'][1])* \
				(gamma_offset+current_gamma)/\
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

				#Logging purposes
				all_requests[i]['RIDE_SHARING_BIAS'][current_gamma] = prob_threshold

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
			
				#Logging purposes
				all_requests[i]['RIDE_SHARING_BIAS'][current_gamma] = prob_threshold


		previous_gamma = current_gamma


	instance['instance_params']['PROB_PARAM_MARKET_SHARE_RIDE_SHARE_INTERNAL'] = coin_flip_params['PROB_PARAM_MARKET_SHARE_RIDE_SHARE_INTERNAL']
	instance['instance_params']['PROB_PARAM_MARKET_SHARE_RIDE_SHARE_EXTERNAL'] = coin_flip_params['PROB_PARAM_MARKET_SHARE_RIDE_SHARE_EXTERNAL']

	instance['all_requests'] = all_requests
	return instance

#helper
def euclidean(x,y):
	assert x is not None and y is not None
	return numpy.linalg.norm(numpy.asarray(x) - numpy.asarray(y))

#helper
def haversine(x,y):

	lon1 = x[0]
	lat1 = x[1]
	lon2 = y[0]
	lat2 = y[1]

	"""
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
	# convert decimal degrees to radians 
	lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
	# haversine formula 
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
	c = 2 * math.asin(math.sqrt(a)) 
	km = 6367 * c + 0.0001
	return km

def get_driving_distance(focus_request,selected_requests,permutation,instance):

	if instance['instance_params']['flag_nyc_data']==True:
		dist = haversine
	else:
		dist = euclidean

	#local
	all_requests = instance['all_requests']
	all_permutations_two = instance['instance_params']['all_permutations_two']

	#Logic
	if len(selected_requests) > 2:
		return NotImplementedError

	if len(selected_requests) == 1: #redundant
		return dist(all_requests[selected_requests[0]]['orig'],all_requests[selected_requests[0]]['dest'])

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
				return dist(all_requests[i]['orig'],all_requests[j]['orig']) + \
						dist(all_requests[j]['orig'],all_requests[i]['dest']) + \
						dist(all_requests[i]['dest'],all_requests[j]['dest'])

			elif focus_request == 'i':
				return dist(all_requests[i]['orig'],all_requests[j]['orig']) + \
						dist(all_requests[j]['orig'],all_requests[i]['dest'])

			elif focus_request == 'j':
				return dist(all_requests[j]['orig'],all_requests[i]['dest']) + \
						dist(all_requests[i]['dest'],all_requests[j]['dest'])
			else:
				raise Exception


		elif all_permutations_two[permutation] == 'case2':

			if focus_request == None or focus_request == 'i':

				return dist(all_requests[i]['orig'],all_requests[j]['orig']) + \
						dist(all_requests[j]['orig'],all_requests[j]['dest']) + \
						dist(all_requests[j]['dest'],all_requests[i]['dest'])
			elif focus_request == 'j':
				return dist(all_requests[j]['orig'],all_requests[j]['dest'])
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

	if instance['instance_params']['flag_nyc_data']==True:
		dist = haversine
	else:
		dist = euclidean

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

			constraint_i =  (all_requests[i]['our_cut_from_requester'][1] - all_requests[i]['our_cut_from_requester'][2])*dist(all_requests[i]['orig'],all_requests[i]['dest']) \
				>= (experiment_params['GAMMA'] + all_requests[i]['detour_sensitivity_normalized'])*\
					(dist(all_requests[i]['orig'],all_requests[j]['orig']) + \
					dist(all_requests[j]['orig'],all_requests[i]['dest']) - \
					dist(all_requests[i]['orig'],all_requests[i]['dest']))
			constraint_j = (all_requests[j]['our_cut_from_requester'][1] - all_requests[j]['our_cut_from_requester'][2])*dist(all_requests[j]['orig'],all_requests[j]['dest']) \
				>= (experiment_params['GAMMA'] + all_requests[j]['detour_sensitivity_normalized'])*\
					(dist(all_requests[j]['orig'],all_requests[i]['dest']) + \
					dist(all_requests[i]['dest'],all_requests[j]['dest']) - \
					dist(all_requests[j]['orig'],all_requests[j]['dest']))

		elif all_permutations_two[permutation] == 'case2':

			constraint_i = (all_requests[i]['our_cut_from_requester'][1] - all_requests[i]['our_cut_from_requester'][2])*dist(all_requests[i]['orig'],all_requests[i]['dest']) \
				>= (experiment_params['GAMMA'] + all_requests[i]['detour_sensitivity_normalized'])*\
					(dist(all_requests[i]['orig'],all_requests[j]['orig']) + \
					dist(all_requests[j]['orig'],all_requests[j]['dest']) + \
					dist(all_requests[j]['dest'],all_requests[i]['dest']) - \
					dist(all_requests[i]['orig'],all_requests[i]['dest']))
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

	if instance['instance_params']['flag_nyc_data']==True:
		dist = haversine
	else:
		dist = euclidean

	#print 'selected_request_as_list',selected_request_as_list

	return instance['instance_params']['ALPHA_OP']*\
		instance['instance_params']['OUR_CUT_FROM_DRIVER']*\
				dist(instance['all_requests'][selected_request_as_list[0]]['orig'],instance['all_requests'][selected_request_as_list[0]]['dest'])

def get_profit_unmatched(selected_request_as_list,instance):
	assert instance is not None
	assert selected_request_as_list is not None	
	#single id is input as a list to maintain API consistency with get_profit() function

	if instance['instance_params']['flag_nyc_data']==True:
		dist = haversine
	else:
		dist = euclidean

	return instance['instance_params']['ALPHA_OP']*\
				(instance['all_requests'][selected_request_as_list[0]]['our_cut_from_requester'][1] - \
					(1 - instance['instance_params']['OUR_CUT_FROM_DRIVER']))*\
				dist(instance['all_requests'][selected_request_as_list[0]]['orig'],instance['all_requests'][selected_request_as_list[0]]['dest'])

def get_profit_matched(selected_requests,instance,permutation,experiment_params):
	

	if instance['instance_params']['flag_nyc_data']==True:
		dist = haversine
	else:
		dist = euclidean

	assert len(selected_requests) == 2

	instance_params = instance['instance_params']
	all_requests = instance['all_requests']


	if experiment_params['DISCOUNT_SETTING']=='independent':

		return instance_params['ALPHA_OP']*\
				(sum([all_requests[m]['our_cut_from_requester'][2]*\
					dist(all_requests[m]['orig'],all_requests[m]['dest']) for m in selected_requests]) - \
				(1 - instance_params['OUR_CUT_FROM_DRIVER'])*get_driving_distance(None,selected_requests,permutation,instance))
	elif experiment_params['DISCOUNT_SETTING']=='detour_based':

		result = 0
		for m in selected_requests:
			beta2 = all_requests[m]['our_cut_from_requester'][1]* \
				(1 - (get_driving_distance(m,selected_requests,permutation,instance) - \
				dist(all_requests[m]['orig'],all_requests[m]['dest']))*\
				1.0/dist(all_requests[m]['orig'],all_requests[m]['dest']))

			result +=instance_params['ALPHA_OP']*beta2*dist(all_requests[m]['orig'],all_requests[m]['dest'])

		return result - \
			instance_params['ALPHA_OP']*(1 - instance_params['OUR_CUT_FROM_DRIVER'])*get_driving_distance(None,selected_requests,permutation,instance)
	else:
		return NotImplementedError

#to retain the edge between two requests for two request case
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

		profits = numpy.zeros(len(all_permutations_two))
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

#helper
def is_interested_in_ridesharing(i,instance,experiment_params):
	GAMMA = experiment_params['GAMMA']
	if instance['all_requests'][i]['PROVIDER_MARKET'][GAMMA]==True:#maybe redundant
		if instance['all_requests'][i]['RIDE_SHARING'][GAMMA]==True:
			return True
	return False #default

#edmonds
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
	'matched_request_pairs_with_permutations':matched_request_pairs_with_permutations,
	'matching_graph':H}

	# pprint(result)

	return result

def get_profit_from_unmatched_requests(selected_requests,instance,experiment_params):

	if selected_requests is None:
		return 0
	elif len(selected_requests) == 0:
		return 0

	result = 0
	for i in selected_requests:
		if instance['all_requests'][i]['RIDE_SHARING'][experiment_params['GAMMA']]==True and \
			instance['all_requests'][i]['PROVIDER_MARKET'][experiment_params['GAMMA']]==True:#if they are in the provider market and ridesharing
			result += get_profit_unmatched([i],instance)

	return result

def get_profit_from_non_ridesharing_requests(selected_requests,instance,experiment_params):

	if selected_requests is None:
		return 0
	elif len(selected_requests) == 0:
		return 0


	#print 'selected_requests',selected_requests

	result = 0
	for i in selected_requests:
		if instance['all_requests'][i]['PROVIDER_MARKET'][experiment_params['GAMMA']]==True:#if they are in the provider market
			result += get_profit_non_ride_sharing([i],instance) #then it counts for the provider

	return result

def get_profit_from_matched_requests(matched_request_pairs_with_permutations,instance,experiment_params):

	if len(matched_request_pairs_with_permutations) == 0:
		return 0

	result = 0
	for request_pairs in matched_request_pairs_with_permutations:
		result += get_profit_matched(request_pairs,instance,matched_request_pairs_with_permutations[request_pairs],experiment_params)
	return result

#solves the allocation for ride sharing part (profit from matched and unmatched) in the market as well as profits from non ridehsharing requests
def solve_instance(coin_flip_no,instance,experiment_params,coeff_internal):

	# assert coeff_internal == instance['instance_params']['PROB_PARAM_MARKET_SHARE_RIDE_SHARE_INTERNAL'] #to be commented

	solution = match_requests(instance,experiment_params)
	total_profit = \
		get_profit_from_non_ridesharing_requests(solution['non_ridesharing_requests'],instance,experiment_params) + \
		get_profit_from_unmatched_requests(solution['unmatched_requests'],instance,experiment_params) + \
		get_profit_from_matched_requests(solution['matched_request_pairs_with_permutations'],instance,experiment_params)

	print "Coeff {5}: Coin flip {4}: Experiment Gamma = {3}. Total profit: {0}. Size of matching graph: {1}. #Total ridesharers: {2}".format(
		total_profit,
		len(solution['matching_graph'].nodes()),
		len([x for x in instance['all_requests'] if instance['all_requests'][x]['RIDE_SHARING'][experiment_params['GAMMA']]==True]),
		experiment_params['GAMMA'],
		coin_flip_no,
		coeff_internal)

	return solution,total_profit

#helper
def get_stats(coin_flip_no,instance):
	#simple helper function to display
	all_gammas = instance['instance_params']['GAMMA_ARRAY_ALL']

	result = []
	for i in instance['all_requests']:
		if instance['all_requests'][i]['PROVIDER_MARKET']['no_gamma']==True:
			result.append(i)

	result2 = [x for x in instance['all_requests'] if x not in result]

	print "Coeff {0}: Coin flip {1}: Stats: (Provider market share w/o Gamma) requests IN: {2}, OUT: {3}".format(instance['instance_params']['PROB_PARAM_MARKET_SHARE_RIDE_SHARE_INTERNAL'],coin_flip_no,len(result),len(result2))

	# #Requests in provider market initially
	# result = []
	# for gamma in all_gammas:
	# 	counter = 0
	# 	for i in instance['all_requests']:
	# 		if i in result:
	# 			counter += 1
	# 			continue
	# 		if instance['all_requests'][i]['PROVIDER_MARKET']['no_gamma']==True:
	# 			if instance['all_requests'][i]['RIDE_SHARING'][gamma]==True:
	# 				counter +=1
	# 				# print "{0} in ridesharing at gamma = {1}".format(i,gamma)
	# 				result.append(i)
	# 	print "Instance: Initially in market: #People ridesharing at gamma = {0} is {1}".format(gamma,counter)

	
	# #Outside provider market
	# result = []
	# for gamma in all_gammas:
	# 	counter = 0
	# 	for i in instance['all_requests']:
	# 		if i in result:
	# 			counter += 1
	# 			continue
	# 		if instance['all_requests'][i]['PROVIDER_MARKET']['no_gamma']==False:
	# 			if instance['all_requests'][i]['RIDE_SHARING'][gamma]==True:
	# 				counter += 1
	# 				# print "{0} in ridesharing at gamma = {1}".format(i,gamma)
	# 				result.append(i)
	# 	print "Instance: Initially out market: #People ridesharing at gamma = {0} is {1}".format(gamma,counter)


	# for gamma in instance['instance_params']['GAMMA_ARRAY_ALL']:
	# 	print "Instance: No of total potential ridesharers at Gamma = {0}: {1}".format(gamma,len([x for x in instance['all_requests'] if instance['all_requests'][x]['RIDE_SHARING'][gamma]==True]))

	# for gamma in instance['instance_params']['GAMMA_ARRAY_ALL']:
	# 	print "No of potential ridesharers at Gamma = {0}: {1}".format(gamma,len([x for x in instance['all_requests'] if instance['all_requests'][x]['RIDE_SHARING'][gamma]==True and instance['all_requests'][x]['PROVIDER_MARKET'][gamma]==True]))

	data = numpy.zeros((len(instance['all_requests']),len(instance['instance_params']['GAMMA_ARRAY_ALL'])))
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

#helper
def get_coin_flip_biases(GAMMA_ARRAY,instance):
	coin_flip_biases = numpy.zeros((len(GAMMA_ARRAY),len(instance['all_requests'])))
	for idx,gamma in enumerate(GAMMA_ARRAY):
		coin_flip_biases[idx] = numpy.asarray([instance['all_requests'][x]['RIDE_SHARING_BIAS'][gamma] for x in instance['all_requests']])
	return coin_flip_biases

#helper: given a gamma and a realization of coin flips caused by a given coin flip param
def get_people_counts(coin_flip_no,instance,experiment_params):
	experiment_params0 = copy.deepcopy(experiment_params)
	experiment_params0['GAMMA'] = 'no_gamma'
	initial_people = [x for x in instance['all_requests'] if instance['all_requests'][x]['PROVIDER_MARKET']['no_gamma']==True]
	
	total_people = [x for x in instance['all_requests'] if instance['all_requests'][x]['PROVIDER_MARKET'][experiment_params['GAMMA']]==True]

	output = (len(total_people),len(total_people) - len(initial_people),
		1.0*(len(total_people) - len(initial_people))/len(initial_people))
	# print output
	return output

#gets coin flip parameters related to initial market and initial ridesharers
def get_coin_flip_params_wo_gamma():
	
	PROB_PARAM_MARKET_SHARE = .6#300.0/ALPHA_OP

	PROB_PARAM_MARKET_SHARE_RIDE_SHARE_NO_GAMMA = 200

	return {'PROB_PARAM_MARKET_SHARE':PROB_PARAM_MARKET_SHARE,
	'PROB_PARAM_MARKET_SHARE_RIDE_SHARE_NO_GAMMA':PROB_PARAM_MARKET_SHARE_RIDE_SHARE_NO_GAMMA}

#generates coin flip parameters for population not initially rideshairng.
def get_coin_flip_params_w_gamma(coin_flip_params,coeff_internal=100):

	assert coin_flip_params is not None

	PROB_PARAM_MARKET_SHARE_RIDE_SHARE_INTERNAL = coeff_internal

	PROB_PARAM_MARKET_SHARE_RIDE_SHARE_EXTERNAL = 0.5*PROB_PARAM_MARKET_SHARE_RIDE_SHARE_INTERNAL

	GAMMA_OFFSET = 0.01

	coin_flip_params.update({
		'PROB_PARAM_MARKET_SHARE_RIDE_SHARE_INTERNAL':PROB_PARAM_MARKET_SHARE_RIDE_SHARE_INTERNAL,
		'PROB_PARAM_MARKET_SHARE_RIDE_SHARE_EXTERNAL': PROB_PARAM_MARKET_SHARE_RIDE_SHARE_EXTERNAL,
		'GAMMA_OFFSET': GAMMA_OFFSET})
	return coin_flip_params


# vprof -c cmh -s profit_maximization.py
if __name__=='__main__':


	flag_nyc_data 	= False
	no_INSTANCES  	= 4
	COEFF_ARRAY_INTERNAL_COINS = [300,400,500,600,700,1e3] # [300,400] # 
	#Above depends on scales of beta,gamma and detour sensitivity
	no_COIN_FLIPS 	= 10 # 100 #
	do_solve 		= True
	GAMMA_ARRAY 	= [0.05,.1,.3,.5,.7,.9] # [.3,.6] #
	instance_params = get_instance_params(GAMMA_ARRAY=GAMMA_ARRAY,flag_nyc_data=flag_nyc_data)
	GAMMA_ARRAY_ALL = instance_params['GAMMA_ARRAY_ALL']
	flag_dump_data  = True

	# Read NYC data from disk
	if flag_nyc_data==True:
		nyc_df = load_nyc_data()
		assert no_INSTANCES < 60 #for the 0 to59 minute blocks in our NYC data.
	else:
		nyc_df = None

	data_multiple_instances = {}
	for instance_no in range(no_INSTANCES):
		print 'Instance {0}: Time : {1}'.format(instance_no,time.ctime())

		instance_base = generate_base_instance(instance_params,flag_nyc_data,instance_no,nyc_df) #no market assignment yet
		coin_flip_params_wo_gamma = get_coin_flip_params_wo_gamma()
		instance_partial = flip_coins_wo_gamma(instance_base,coin_flip_params_wo_gamma) #to keep market share and initial division in market share the same as this stochasticity need not be averaged.

		#noSIR. Hence, does not depend on probability coefficient or gamma values
		experiment_params = {'DISCOUNT_SETTING':'detour_based','GAMMA':'no_gamma'} # 'independent'
		baseline_solution,baseline_profit = solve_instance(-1,instance_partial,experiment_params,-1)
		baseline_instance = instance_partial

		#with SIR
		profits_given_coeffs  = {} #logging purposes
		coin_flip_params_dict = {} #logging purposes
		coin_flip_biases_dict = {} #logging purposes
		for coeff_no,coeff_internal in enumerate(COEFF_ARRAY_INTERNAL_COINS):

			coin_flip_params = get_coin_flip_params_w_gamma(coin_flip_params_wo_gamma,coeff_internal)

			instance_dict = {}
			solution_dict = {}
			total_profit_array 							= numpy.zeros((len(GAMMA_ARRAY),no_COIN_FLIPS))
			people_count_dict = {}
			people_count_dict['total_people'] 			= numpy.zeros((len(GAMMA_ARRAY),no_COIN_FLIPS))
			people_count_dict['additional_people'] 		= numpy.zeros((len(GAMMA_ARRAY),no_COIN_FLIPS))
			people_count_dict['additional_people_pct'] 	= numpy.zeros((len(GAMMA_ARRAY),no_COIN_FLIPS))
			for coin_flip_no in range(no_COIN_FLIPS):
				instance = flip_coins_w_gamma(instance_partial,coin_flip_params) #only influx from internal and external changes
				print 'Coeff {2}: Coin flip {0}: Starting corresponding experiment: Time {1}'.format(coin_flip_no,time.ctime(),coeff_internal)
				#get_stats(coin_flip_no,instance)

				if do_solve is True:
					for idx,gamma in enumerate(GAMMA_ARRAY):
						experiment_params = {'DISCOUNT_SETTING':'detour_based','GAMMA':gamma} # 'independent'
						solution,total_profit = solve_instance(coin_flip_no,instance,experiment_params,coeff_internal)

						#Logging
						total_profit_array[idx,coin_flip_no] = total_profit
						solution_dict[(idx,coin_flip_no)] = solution
						(people_count_dict['total_people'][idx,coin_flip_no],\
						people_count_dict['additional_people'][idx,coin_flip_no],\
						people_count_dict['additional_people_pct'][idx,coin_flip_no]) = get_people_counts(coin_flip_no,instance,experiment_params)
						instance_dict[(idx,coin_flip_no)] = instance
			
			#Logging
			coin_flip_params_dict[coeff_no] = coin_flip_params #logging purposes		
			coin_flip_biases_dict[coeff_no] = get_coin_flip_biases(GAMMA_ARRAY,instance) #note we assume the last coin flip instance is available outside the no_COIN_FLIPS loop
			profits_given_coeffs[coeff_no] = \
				{'total_profit_array'	: total_profit_array,
				 'solution_dict'		: solution_dict,
				 'people_count_dict'	: people_count_dict,
				 'instance_dict'		:instance_dict,
				 'baseline_instance': baseline_instance,
				 'baseline_profit': baseline_profit,
				 'baseline_solution': baseline_solution}

		print '\n\n\n Instance {0} completed.'.format(instance_no)

		data_multiple_instances[instance_no] = {'profits_given_coeffs':profits_given_coeffs,
			'instance_base':instance_base,
			'COEFF_ARRAY_INTERNAL_COINS':COEFF_ARRAY_INTERNAL_COINS,
			'no_COIN_FLIPS':no_COIN_FLIPS,
			'coin_flip_biases_dict':coin_flip_biases_dict,
			'coin_flip_params_dict':coin_flip_params_dict}	

	print 'Ending all experiments: The time is :', time.ctime()

	if flag_dump_data:
		pickle.dump(data_multiple_instances,
			open('../../../Xharecost_MS_annex/plot_data.pkl','wb'))