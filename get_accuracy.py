import numpy as np
import sys

file = str(sys.argv[1])
train = np.array([])
test = np.array([])
testing = False
with open(file) as f:
    for line in f:
        if 'Testing net ' in line:
            testing = True
        if 'accuracy = ' in line:
            if testing:
                test = np.append(test, float(line.split('= ')[1]))
                testing = False
            else:
                train = np.append(train, float(line.split('= ')[1]))

import cPickle
ext_array = (file.split('.')[0]).split('_')[-3:]
with open('accuracy_' + "_".join(ext_array) + '.pkl', 'wb') as f:
#with open('accuracy_' + (file.split('_')[-2]) + '_' +  (file.split('_')[-1]).split('.')[0] + '.pkl', 'wb') as f:
    cPickle.dump({'train': train, 'test': test}, f)
