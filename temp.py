#TODO:

edit plot_probability_gamma(data)




#######################################################################
import pickle, numpy
data_multiple_instances = pickle.load(open('../../../Xharecost_MS_annex/plot_data.pkl','rb'))
print multiple_instances_data[0]['instance_base']['instance_params']['GAMMA_ARRAY']
print multiple_instances_data[0]['instance_base']['instance_params']['GAMMA_ARRAY_ALL']
print [x for x in data_multiple_instances]
print [x for x in data_multiple_instances[0]]
print [x for x in data_multiple_instances[0]['profits_given_coeffs']]
print [x for x in data_multiple_instances[0]['profits_given_coeffs'][0]]
print data_multiple_instances[0]['profits_given_coeffs'][0]['baseline_profit']
temp = data_multiple_instances[0]['profits_given_coeffs'][0]['total_profit_array']
print temp
print temp - 40000
print temp/1000
print numpy.zeros(temp.shape)
#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################
