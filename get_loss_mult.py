import numpy as np
import sys

files = sys.argv[1:]
train = np.array([])
test = np.array([])
for file in files:
    with open(file) as f:
        for line in f:
            if 'Test net output' in line:
                test = np.append(test, float( (line.split('= ')[1]).split(' (')[0] ))
            elif 'Train net output' in line:
                train = np.append(train, float( (line.split('= ')[1]).split(' (')[0] ))

import cPickle
#with open('accuracy_' + (file.split('-')[1]).split('.')[0] + '.pkl', 'wb') as f:
#with open('accuracy_' + (file.split('_')[-2]) + '_' +  (file.split('_')[-1]).split('.')[0] + '.pkl', 'wb') as f:
ext_array = ((file.split('.')[0]).split('/')[1]).split('_')[-4:]
with open('accuracy_' + "_".join(ext_array) + '.pkl', 'wb') as f:
    cPickle.dump({'train': train, 'test': test}, f)
