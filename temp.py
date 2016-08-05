#TODO:


#######################################################################

import matplotlib.pyplot as plt


x = [instance['all_requests'][z]['orig'][0] for z in instance['all_requests']]
y = [instance['all_requests'][z]['orig'][1] for z in instance['all_requests']]

plt.subplot(111)
plt.scatter(x, y, s=80, c=range(len(instance['all_requests'])), marker="o")

plt.show()

#######################################################################

import matplotlib.pyplot as plt

plt.subplot(111)

pd = {'orig':'green','dest':'red'}
for loc in pd:
    x = [instance['all_requests'][z][loc][0] for z in instance['all_requests']]
    y = [instance['all_requests'][z][loc][1] for z in instance['all_requests']]

    plt.scatter(x, y, s=80, c=pd[loc], marker="o")

plt.show()






#######################################################################




def get_driving_distance_old(selected_requests,permutation,instance):

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
		if permutation[0]=='sj':#i and j are symmetric within each case. See board photo. TODO
			(i,j) = (j,i)

		if all_permutations_two[permutation] == 'case1':
			
			return euclidean(all_requests[i]['orig'],all_requests[j]['orig']) + \
					euclidean(all_requests[j]['orig'],all_requests[i]['dest']) + \
					euclidean(all_requests[i]['dest'],all_requests[j]['dest'])

		elif all_permutations_two[permutation] == 'case2':

			return euclidean(all_requests[i]['orig'],all_requests[j]['orig']) + \
					euclidean(all_requests[j]['orig'],all_requests[j]['dest']) + \
					euclidean(all_requests[j]['dest'],all_requests[i]['dest'])
		else:
			print "Error!" #TODO
			return -1 #TODO


#######################################################################



	#permutations needed for two participant matching
	all_permutations_two = {('si','sj','di','dj'):'case1',
			('si','sj','dj','di'):'case2',
			('sj','si','dj','di'):'case1',
			('sj','si','di','dj'):'case2'}#TODO: make more comprehensible

#######################################################################


import matplotlib.pyplot as plt


N = len(instance['all_requests'])
labels = ['{0}'.format(i) for i in range(N)]
plt.subplots_adjust(bottom = 0.1)
pd = {'orig':'green','dest':'red'}
for loc in pd:
    x = [instance['all_requests'][z][loc][0] for z in instance['all_requests']]
    y = [instance['all_requests'][z][loc][1] for z in instance['all_requests']]

#     plt.scatter(x, y, s=80, c=pd[loc], marker="o")

    plt.scatter(x, y, marker = 'o', 
        c = pd[loc], s = 1500,
        cmap = plt.get_cmap('Spectral'))
    for label, x, y in zip(labels, x, y):
        plt.annotate(
            label+','+loc, 
            xy = (x, y), xytext = (-20, 20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

plt.show()

#######################################################################




#######################################################################
