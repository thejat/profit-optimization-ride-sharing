import numpy, math, pickle, copy
import matplotlib
import matplotlib.pyplot as plt

def get_data(filepath='../../../../sharecost_ms_annex/plot_data_test.pkl'):

	print "DEFAULT IS TEST DATA UNLESS SPECIFIED OTHERWISE"

	#read data
	data_multiple_instances = pickle.load(open(filepath,'rb'))
	no_INSTANCES = len([x for x in data_multiple_instances])
	no_COIN_FLIPS = data_multiple_instances[0]['no_COIN_FLIPS']

	#create processed data for plotters
	data = {}
	data['COEFF_ARRAY_INTERNAL_COINS'] = data_multiple_instances[0]['COEFF_ARRAY_INTERNAL_COINS'] 
	data['GAMMA_ARRAY'] = data_multiple_instances[0]['instance_base']['instance_params']['GAMMA_ARRAY']
	data['baseline_profit'] = 0


	#profit as a function of gamma
	data['profit_by_gamma'] = {}
	for coeff_no,coeff_internal in enumerate(data['COEFF_ARRAY_INTERNAL_COINS']):
		
		for instance_no in data_multiple_instances:
			temp = (data_multiple_instances[instance_no]['profits_given_coeffs'][coeff_no]['total_profit_array'] \
				  - data_multiple_instances[instance_no]['profits_given_coeffs'][coeff_no]['baseline_profit']) \
					/(data_multiple_instances[instance_no]['profits_given_coeffs'][coeff_no]['baseline_profit'] + 1e-5)
			if instance_no==0:
				data_matrix = copy.deepcopy(temp)
			else:
				data_matrix = numpy.hstack((data_matrix,temp))
		data['profit_by_gamma'][coeff_no] = {}
		data['profit_by_gamma'][coeff_no]['median'] = numpy.median(data_matrix, axis=1)
		data['profit_by_gamma'][coeff_no]['std'] = numpy.std(data_matrix, axis=1)/math.sqrt(data_matrix.shape[1])


	#profit as a function of prob coeff
	data['profit_by_prob'] = {}
	for gamma_idx,gamma in enumerate(data['GAMMA_ARRAY']):
		data_matrix = numpy.zeros((len(data['COEFF_ARRAY_INTERNAL_COINS']),
			no_COIN_FLIPS*no_INSTANCES))
		for coeff_no,coeff_internal in enumerate(data['COEFF_ARRAY_INTERNAL_COINS']):

			for instance_no in data_multiple_instances:
				temp = (data_multiple_instances[instance_no]['profits_given_coeffs'][coeff_no]['total_profit_array'][gamma_idx,] \
					  - data_multiple_instances[instance_no]['profits_given_coeffs'][coeff_no]['baseline_profit']) \
						/(data_multiple_instances[instance_no]['profits_given_coeffs'][coeff_no]['baseline_profit'] + 1e-5)
				if instance_no ==0:
					data_vector = copy.deepcopy(temp)
				else:
					data_vector = numpy.hstack((data_vector,temp))
			data_matrix[coeff_no,] = data_vector

		data['profit_by_prob'][gamma_idx] = {}
		data['profit_by_prob'][gamma_idx]['median'] = numpy.median(data_matrix, axis=1)
		data['profit_by_prob'][gamma_idx]['std'] = numpy.std(data_matrix, axis=1)/math.sqrt(data_matrix.shape[1])

	#additional people as a function of gamma
	data['people_by_gamma'] = {}
	for coeff_no,coeff_internal in enumerate(data['COEFF_ARRAY_INTERNAL_COINS']):
		
		for instance_no in data_multiple_instances:
			temp = data_multiple_instances[instance_no]['profits_given_coeffs'][coeff_no]['people_count_dict']['additional_people_pct']
			if instance_no==0:
				data_matrix = copy.deepcopy(temp)
			else:
				data_matrix = numpy.hstack((data_matrix,temp))
		data['people_by_gamma'][coeff_no] = {}
		data['people_by_gamma'][coeff_no]['median'] = numpy.median(data_matrix, axis=1)
		data['people_by_gamma'][coeff_no]['std'] = numpy.std(data_matrix, axis=1)/math.sqrt(data_matrix.shape[1])



	#A 2d array of percentage increase in profit
	deltaP = numpy.zeros((len(data['GAMMA_ARRAY']),len(data['COEFF_ARRAY_INTERNAL_COINS'])))
	for x,v in enumerate(data['GAMMA_ARRAY']):
		for y,v2 in enumerate(data['COEFF_ARRAY_INTERNAL_COINS']):
			deltaP[x,y] = data['profit_by_gamma'][y]['median'][x]

	#A 2d array of percentage increase in people/marketshare
	deltaN = numpy.zeros((len(data['GAMMA_ARRAY']),len(data['COEFF_ARRAY_INTERNAL_COINS'])))
	for x,v in enumerate(data['GAMMA_ARRAY']):
		for y,v2 in enumerate(data['COEFF_ARRAY_INTERNAL_COINS']):
			deltaN[x,y] = data['people_by_gamma'][y]['median'][x]

	lb_vector = linspace(0,0.1,3)
	MAX_DELTA_N = numpy.inf
	min_market_share_by_profit_lb = numpy.zeros((len(lb_vector),len(data['GAMMA_ARRAY'])))
	for w,lb in enumerate(lb_vector):
	    for x,v1 in enumerate(data['GAMMA_ARRAY']):
	        min_deltaN = MAX_DELTA_N
	        for y,v2 in enumerate(data['COEFF_ARRAY_INTERNAL_COINS']):
	            if deltaP[x,y] >= lb:
	                min_deltaN = min(min_deltaN,deltaN[x,y])
	        if min_deltaN<MAX_DELTA_N:
	            min_market_share_by_profit_lb[w,x] = min_deltaN
	        else:
	            min_market_share_by_profit_lb[w,x] = None
	                
	#print min_market_share_by_profit_lb
	data['deltaN'] = deltaN
	data['deltaP'] = deltaP
	data['min_market_share_by_profit_lb'] = min_market_share_by_profit_lb
	data['lb_vector'] = lb_vector


	data['data_multiple_instances'] = data_multiple_instances #hack for backward compatibility
	return data

#helper
def linspace(a, b, n=10):
    if n < 2:
        return b
    diff = (float(b) - a)/(n - 1)
    return [diff * i + a  for i in range(n)]

#profit increase vs marketshare increase
def intro_plot(data):
#     profit_levels = linspace(min(data['deltaP'].ravel()),max(data['deltaP'].ravel()),10)
    market_share_levels = linspace(min(data['deltaN'].ravel()),max(data['deltaN'].ravel()),10)
    MIN_DELTA_P = -1*numpy.inf
    max_profit_by_max_market_share = numpy.zeros(len(market_share_levels))
    for w,ub in enumerate(market_share_levels):
        profit_level = MIN_DELTA_P
        for i in range(data['deltaN'].shape[0]):
            for j in range(data['deltaN'].shape[1]):
                if data['deltaN'][i,j] <= ub:
                    profit_level = max(profit_level,data['deltaP'][i,j])
        if profit_level > MIN_DELTA_P:
            max_profit_by_max_market_share[w] = profit_level
        else:
            max_profit_by_max_market_share[w] = None
    #data['max_profit_by_max_market_share'] = max_profit_by_max_market_share
    #data['market_share_levels'] = market_share_levels
    #return [max_profit_by_max_market_share,market_share_levels]

    import matplotlib
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = market_share_levels
    ys = max_profit_by_max_market_share
    ax.plot(xs, ys)
    ys0 = numpy.zeros(len(max_profit_by_max_market_share))
    ax.plot(xs, ys0)
    plt.xlabel('Pct increase in marketshare due to SIR (cumulative)')
    plt.ylabel('Max pct increase in profit given marketshare UB')
    plt.show()

#plot minimum marketshare for profit lowerbound
def plot_ms_vs_profit_lb(data,lb_vector=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #code repetition
    if lb_vector is not None:
		MAX_DELTA_N = numpy.inf
		min_market_share_by_profit_lb = numpy.zeros((len(lb_vector),len(data['GAMMA_ARRAY'])))
		for w,lb in enumerate(lb_vector):
		    for x,v1 in enumerate(data['GAMMA_ARRAY']):
		        min_deltaN = MAX_DELTA_N
		        for y,v2 in enumerate(data['COEFF_ARRAY_INTERNAL_COINS']):
		            if data['deltaP'][x,y] >= lb:
		                min_deltaN = min(min_deltaN,data['deltaN'][x,y])
		        if min_deltaN<MAX_DELTA_N:
		            min_market_share_by_profit_lb[w,x] = min_deltaN
		        else:
		            min_market_share_by_profit_lb[w,x] = None
		                
		#print min_market_share_by_profit_lb
		data['min_market_share_by_profit_lb'] = min_market_share_by_profit_lb
		data['lb_vector'] = lb_vector
	#code repetition ends


    for lb_idx,lb in enumerate(data['lb_vector']):
        xs 		= data['GAMMA_ARRAY']
        ys 		= data['min_market_share_by_profit_lb'][lb_idx,]
        ax.plot(xs, ys,label=lb)

    legend = ax.legend(loc='best', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    plt.xlabel('Gamma')
    plt.ylabel('Min Marketshare increase in pct given profit LB')
    plt.show()

#plot profit as a function of gamma
def plot_profit_vs_gamma(data):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	#w Gamma
	for coeff_no,coeff_internal in enumerate(data['COEFF_ARRAY_INTERNAL_COINS']):

		xs 		= data['GAMMA_ARRAY']
		ys 		= data['profit_by_gamma'][coeff_no]['median']
		ys_std 	= data['profit_by_gamma'][coeff_no]['std']

		ax.fill_between(xs, ys-ys_std, ys+ys_std, 
			alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
		ax.plot(xs, ys,label=coeff_internal)

	#w/o Gamma, will be the same. assumed here, not asserted.
	ys	= numpy.asarray([data['baseline_profit'] for x in range(len(ys))]) #replicate
	ax.plot(xs, ys,label='No SIR')


	legend = ax.legend(loc='best', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	for label in legend.get_texts():
		label.set_fontsize('large')
	for label in legend.get_lines():
		label.set_linewidth(1.5)  # the legend line width

	plt.xlabel('Gamma')
	plt.ylabel('Profit')
	plt.title('Variation of profit with gamma')
	plt.show()	

#plot profit as a function of probability coefficient
def plot_profit_vs_probability(data):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	#with SIR
	xs 		= data['COEFF_ARRAY_INTERNAL_COINS']
	for gamma_idx,gamma in enumerate(data['GAMMA_ARRAY']):

		ys 		= data['profit_by_prob'][gamma_idx]['median']
		ys_std 	= data['profit_by_prob'][gamma_idx]['std']
 
		ax.fill_between(xs, ys-ys_std, ys+ys_std, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
		ax.plot(xs, ys,label=gamma)

	#without SIR
	ys	= numpy.asarray([data['baseline_profit'] for x in range(len(ys))]) #replicate
	ax.plot(xs, ys,label='No SIR')

	legend = ax.legend(loc='best', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	# Set the fontsize
	for label in legend.get_texts():
		label.set_fontsize('large')
	for label in legend.get_lines():
		label.set_linewidth(1.5)  # the legend line width

	#plt.xscale('log')
	plt.xlabel('Probability coefficient')
	plt.ylabel('Profit')
	plt.title('Profit vs probability of choosing to rideshare')
	plt.show()

#plots median probability values as a function of gamma.
def plot_probability_gamma(data):
	return NotImplementedError
	# do not store instances, hence the code fails
	# also, averaging across instances not done

	fig = plt.figure()
	ax = fig.add_subplot(111)

	data = data['data_multiple_instances'][0]

	COEFF_ARRAY_INTERNAL_COINS = data['COEFF_ARRAY_INTERNAL_COINS']
	coin_flip_params_dict = data['coin_flip_params_dict']
	profits_given_coeffs = data['profits_given_coeffs']
	no_COIN_FLIPS = data['no_COIN_FLIPS']
	instance_base = data['instance_base']
	coin_flip_biases_dict = data['coin_flip_biases_dict']
	coeff_indices = [x for x in data['profits_given_coeffs']]
	xticklabels = data['instance_base']['instance_params']['GAMMA_ARRAY']

	output = numpy.zeros((len(COEFF_ARRAY_INTERNAL_COINS),len(instance_base['instance_params']['GAMMA_ARRAY'])))
	output_std = numpy.zeros((len(COEFF_ARRAY_INTERNAL_COINS),len(instance_base['instance_params']['GAMMA_ARRAY'])))
	for coeff_no,coeff_internal in enumerate(COEFF_ARRAY_INTERNAL_COINS):
	    for idx,gamma in enumerate(instance_base['instance_params']['GAMMA_ARRAY']):
	        temp_meta = []
	        for coin_flip_no in range(no_COIN_FLIPS):
	            temp = []
	            for i in instance_base['all_requests']:
	                #CONDITIONING ON BEING OUTSIDE MARKETSHARE
	                if profits_given_coeffs[coeff_no]['instance_dict'][(idx,coin_flip_no)]['all_requests'][i]['PROVIDER_MARKET']['no_gamma']==False:
	                    temp.append(coin_flip_biases_dict[coeff_no][idx,i])
	            temp_meta.append(numpy.median(numpy.asarray(temp)))
	        output[coeff_no,idx] = numpy.median(numpy.asarray(temp_meta))
	        output_std[coeff_no,idx] = numpy.std(numpy.asarray(temp_meta))/math.sqrt(len(temp_meta))
	print output

	for coeff_index in coeff_indices:
		
		temp_avg = output[coeff_index,:]
		#print temp_avg
		temp_std = output_std[coeff_index,:]

		xs 		= xticklabels
		ys 		= temp_avg #exclude no_gamma
		ys_std 	= temp_std #exclude no_gamma

		ax.fill_between(xs, ys-ys_std, ys+ys_std, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
		ax.plot(xs, ys,label=data['COEFF_ARRAY_INTERNAL_COINS'][coeff_index])

	legend = ax.legend(loc='best', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	# Set the fontsize
	for label in legend.get_texts():
		label.set_fontsize('large')
	for label in legend.get_lines():
		label.set_linewidth(1.5)  # the legend line width
	plt.xlabel('Gamma')
	plt.ylabel('Probability')
	plt.title('Probability of Conversion (outside provider\'s market)')
	plt.show()	

#needs a single experiment solution and its corresponding instance. Will plot the OD and the rideshares that happen. Used in single_experiment.ipynb
def plot_OD_ridesharing(solution,instance):
	plot_rs_matched = True
	if plot_rs_matched == True:
	    temp_request_set = set()
	    for k in solution['matched_request_pairs_with_permutations']:
	        temp_request_set.add(k[0])
	        temp_request_set.add(k[1])
	    data = {x:instance['all_requests'][x] for x in temp_request_set}
	else:        
	    data = {x:instance['all_requests'][x] for x in instance['all_requests'] if instance['all_requests'][x]['RIDE_SHARING'][experiment_params['GAMMA']]==True}

	N = len(data)
	labels = ['{0}'.format(i) for i in data]

	fig = plt.figure()
	ax = fig.add_subplot(111)

	pd = {'orig':'green','dest':'red'}
	for loc in pd:
	    x = [data[z][loc][0] for z in data]
	    y = [data[z][loc][1] for z in data]

	#     plt.scatter(x, y, s=80, c=pd[loc], marker="o")

	    ax.scatter(x, y, marker = 'o', 
	        c = pd[loc], s = 1500,
	        cmap = plt.get_cmap('Spectral'))
	    for label, x, y in zip(labels, x, y):
	        ax.annotate(
	            label+','+loc, 
	            xy = (x, y), xytext = (-20, 20),
	            textcoords = 'offset points', ha = 'right', va = 'bottom',
	            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
	            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

	data_matched = solution['matched_request_pairs_with_permutations']
	for p,q in data_matched:
	    if instance['instance_params']['all_permutations_two'][data_matched[(p,q)]]=='case1':
	        print "case1: {0},{1}:{2}".format(p,q,data_matched[(p,q)])
	        if data_matched[(p,q)][0] == 'sj':
	            (p,q) = (q,p)
	        ax.plot([data[p]['orig'][0],data[q]['orig'][0],data[p]['dest'][0],data[q]['dest'][0]],\
	                [data[p]['orig'][1],data[q]['orig'][1],data[p]['dest'][1],data[q]['dest'][1]],linewidth=6)
	    elif instance['instance_params']['all_permutations_two'][data_matched[(p,q)]]=='case2':
	        print "case2: {0},{1}:{2}".format(p,q,data_matched[(p,q)])
	        if data_matched[(p,q)][0] == 'sj':
	            (p,q) = (q,p)
	        ax.plot([data[p]['orig'][0],data[q]['orig'][0],data[q]['dest'][0],data[p]['dest'][0]],\
	                [data[p]['orig'][1],data[q]['orig'][1],data[q]['dest'][1],data[p]['dest'][1]],linewidth=6)

	plt.xlabel('X coordinate')
	plt.ylabel('Y coordinate')
	plt.title('OD Spatial Plot')
	plt.show()