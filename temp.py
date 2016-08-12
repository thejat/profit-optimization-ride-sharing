#TODO:

# edit plot_probability_gamma(data)




#######################################################################
# import pickle, numpy
# data_multiple_instances = pickle.load(open('../../../Xharecost_MS_annex/plot_data.pkl','rb'))
# print multiple_instances_data[0]['instance_base']['instance_params']['GAMMA_ARRAY']
# print multiple_instances_data[0]['instance_base']['instance_params']['GAMMA_ARRAY_ALL']
# print [x for x in data_multiple_instances]
# print [x for x in data_multiple_instances[0]]
# print [x for x in data_multiple_instances[0]['profits_given_coeffs']]
# print [x for x in data_multiple_instances[0]['profits_given_coeffs'][0]]
# print data_multiple_instances[0]['profits_given_coeffs'][0]['baseline_profit']
# temp = data_multiple_instances[0]['profits_given_coeffs'][0]['total_profit_array']
# print temp
# print temp - 40000
# print temp/1000
# print numpy.zeros(temp.shape)
#######################################################################


import multiprocessing

def funSquare(num):
    return num ** 2

if __name__ == '__main__':
    pool = multiprocessing.Pool()
    results = pool.map(funSquare, range(10))
    print(results)

#######################################################################

%matplotlib notebook
import numpy, pickle
#data = pickle.load(open('../../../Xharecost_MS_annex/plot_data_DELETE.pkl','rb'))
#print "DELETE VERSION"
data = pickle.load(open('../../../Xharecost_MS_annex/plot_data.pkl','rb'))
COEFF_ARRAY_INTERNAL_COINS = data['COEFF_ARRAY_INTERNAL_COINS']
coin_flip_params_dict = data['coin_flip_params_dict']
profits_given_coeffs = data['profits_given_coeffs']
no_COIN_FLIPS = data['no_COIN_FLIPS']
instance_base = data['instance_base']
coin_flip_biases_dict = data['coin_flip_biases_dict']
#Median coin flip probabilities
for coeff_no,coeff_internal in enumerate(COEFF_ARRAY_INTERNAL_COINS):
    #print profits_given_coeffs[coeff_no]['people_count_dict']['additional_people']
    print numpy.median(coin_flip_biases_dict[coeff_no],axis=1)
    #Median coin flip probabilities of initial-outside-market-requests
output = numpy.zeros((len(COEFF_ARRAY_INTERNAL_COINS),len(instance_base['instance_params']['GAMMA_ARRAY'])))
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
print output
#######################################################################
#######################################################################
#######################################################################
