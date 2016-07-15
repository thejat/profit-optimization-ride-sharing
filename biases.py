import numpy
from profit_maximization import *

if __name__=='__main__':


	COEFF_ARRAY_INTERNAL_COINS = [100,1000] #Depends on scales of beta,gamma and detour sensitivity

	instance_base = generate_base_instance() #no market assignment yet
	coin_flip_params_wo_gamma = get_coin_flip_params_wo_gamma()
	instance_partial = flip_coins_wo_gamma(instance_base,coin_flip_params_wo_gamma)

	instances = {}
	for coeff_no,coeff_internal in enumerate(COEFF_ARRAY_INTERNAL_COINS):

		coin_flip_params = get_coin_flip_params_w_gamma(coin_flip_params_wo_gamma,coeff_internal)
	
		instance = flip_coins_w_gamma(instance_partial,coin_flip_params)
		instances[coeff_internal] = instance

	pickle.dump({'instances':instances,\
		'COEFF_ARRAY_INTERNAL_COINS':COEFF_ARRAY_INTERNAL_COINS,
		'coin_flip_params':coin_flip_params},
		open('../../../Xharecost_MS_annex/bias_data.pkl','wb'))	