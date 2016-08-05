import numpy
from profit_maximization import *
from collections import OrderedDict

#Compute coin flip parameter values for each gamma and coefficient value

if __name__=='__main__':

	no_INSTANCES  	= 1
	COEFF_ARRAY_INTERNAL_COINS = [50,150,250,350,1e3] #Depends on scales of beta,gamma and detour sensitivity
	GAMMA_ARRAY 	= [0,.1,.3,.5,.7,.9]
	instance_params = get_instance_params(GAMMA_ARRAY=GAMMA_ARRAY)
	GAMMA_ARRAY_ALL = instance_params['GAMMA_ARRAY_ALL']
	
	for i in range(no_INSTANCES):
		print 'Instance {0}: Time : {1}'.format(i,time.ctime())

		instance_base = generate_base_instance(instance_params) #no market assignment yet
		coin_flip_params_wo_gamma = get_coin_flip_params_wo_gamma()
		instance_partial = flip_coins_wo_gamma(instance_base,coin_flip_params_wo_gamma) #to keep market share and initial division in market share the same as this stochasticity need not be averaged.

		instances = {}
		coin_flip_params_dict = {}
		all_biases = {}
		for coeff_no,coeff_internal in enumerate(COEFF_ARRAY_INTERNAL_COINS):

			coin_flip_params = get_coin_flip_params_w_gamma(coin_flip_params_wo_gamma,coeff_internal)
			
			instance = flip_coins_w_gamma(instance_partial,coin_flip_params) #only influx from internal and external changes

			instances[coeff_internal] = instance



	pickle.dump({'instances':instances,\
		'COEFF_ARRAY_INTERNAL_COINS':COEFF_ARRAY_INTERNAL_COINS,
		'coin_flip_params_dict':coin_flip_params_dict},
		'all_biases':all_biases,
		'GAMMA_ARRAY':GAMMA_ARRAY,
		open('../../../Xharecost_MS_annex/bias_data.pkl','wb'))