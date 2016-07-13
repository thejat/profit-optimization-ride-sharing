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

def plot_result_all(data):
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
