import numpy, math, pickle
import matplotlib
import matplotlib.pyplot as plt


def plot_result_single(data_matrix,xticklabels):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	temp_avg = numpy.median(data_matrix, axis=0)
	#print temp_avg
	temp_std = numpy.std(data_matrix, axis=0)/math.sqrt(data_matrix.shape[0])

	#w Gamma
	xs 		= range(len(xticklabels)-1) #GAMMA_ARRAY_ALL with 'no_gamma' removed
	ys 		= temp_avg[1:] #exclude no_gamma
	ys_std 	= temp_std[1:] #exclude no_gamma

	ax.fill_between(xs, ys-ys_std, ys+ys_std, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
	ax.plot(xs, ys)

	#w/o Gamma
	ys_baselime 	= numpy.asarray([temp_avg[0] for x in range(len(ys))]) #replicate
	ys_std_baseline = numpy.asarray([temp_std[0] for x in range(len(ys))]) #replicate
	ax.fill_between(xs, ys_baselime-ys_std_baseline, ys_baselime+ys_std_baseline, alpha=0.5, edgecolor='#CC4A3B', facecolor='#FFAAAA')
	ax.plot(xs, ys_baselime)


	plt.setp(ax, xticks=xs, xticklabels=xticklabels[1:])
	plt.xlabel('Gamma')
	plt.ylabel('Profit')
	plt.title('Variation of profit with gamma')
	plt.tight_layout()
	plt.ylim((.9*min(ys),1.2*max(ys)))
	plt.show()

def plot_result_gamma(data):
	fig = plt.figure()
	ax = fig.add_subplot(111)


	coeff_indices = [x for x in data['profits_given_coeffs']]
	xticklabels = data['instance_base']['instance_params']['GAMMA_ARRAY_ALL']

	for coeff_index in coeff_indices:
		data_matrix = data['profits_given_coeffs'][coeff_index]['total_profit_array'].transpose()

		temp_avg = numpy.median(data_matrix, axis=0)
		#print temp_avg
		temp_std = numpy.std(data_matrix, axis=0)/math.sqrt(data_matrix.shape[0])

		#w Gamma
		xs 		= range(len(xticklabels)-1) #GAMMA_ARRAY_ALL with 'no_gamma' removed
		ys 		= temp_avg[1:] #exclude no_gamma
		ys_std 	= temp_std[1:] #exclude no_gamma

		ax.fill_between(xs, ys-ys_std, ys+ys_std, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
		ax.plot(xs, ys,label=data['COEFF_ARRAY_INTERNAL_COINS'][coeff_index])

	#w/o Gamma, will be the same. assumed here, not asserted.
	ys_baselime 	= numpy.asarray([temp_avg[0] for x in range(len(ys))]) #replicate
	ys_std_baseline = numpy.asarray([temp_std[0] for x in range(len(ys))]) #replicate
	ax.fill_between(xs, ys_baselime-ys_std_baseline, ys_baselime+ys_std_baseline, alpha=0.5, edgecolor='#CC4A3B', facecolor='#FFAAAA')
	ax.plot(xs, ys_baselime,label='no_gamma')

	legend = ax.legend(loc='best', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	# Set the fontsize
	for label in legend.get_texts():
		label.set_fontsize('large')
	for label in legend.get_lines():
		label.set_linewidth(1.5)  # the legend line width
	plt.setp(ax, xticks=xs, xticklabels=xticklabels[1:])
	plt.xlabel('Gamma')
	plt.ylabel('Profit')
	plt.title('Variation of profit with gamma')
	#plt.tight_layout()
	#plt.ylim((.9*min(ys),1.2*max(ys)))
	plt.show()	

def plot_result_probability(data):
	fig = plt.figure()
	ax = fig.add_subplot(111)

	GAMMA_ARRAY_ALL = data['instance_base']['instance_params']['GAMMA_ARRAY_ALL']
	xticklabels = data['COEFF_ARRAY_INTERNAL_COINS']

	for gamma_idx,gamma in enumerate(GAMMA_ARRAY_ALL):

		data_matrix = numpy.asarray([data['profits_given_coeffs'][coeff_index]['total_profit_array'][gamma_idx,] for coeff_index in range(len(xticklabels))]).transpose()
		# print 'data_matrix',data_matrix

		ys 		= numpy.median(data_matrix, axis=0)
		ys_std 	= numpy.std(data_matrix, axis=0)/math.sqrt(data_matrix.shape[0])
		xs 		= xticklabels 

		# print 'ys',ys

		ax.fill_between(xs, ys-ys_std, ys+ys_std, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
		ax.plot(xs, ys,label=gamma)

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
	plt.title('Variation of profit with probability of choosing to rideshare')
	plt.show()


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