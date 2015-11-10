from rosch_analysis import rosch_analysis
import numpy as np
import yamutils.fast as fast
import cPickle

import dldata.stimulus_sets.synthetic.synthetic_datasets as sd
dataset = sd.RoschDataset3_simple74testFewer()

features = cPickle.load(open('/om/user/hyo/caffe/features/catInet_roschSimple74testFewer_fc7.p'))
perm = np.random.RandomState(0).permutation(len(features))
pinv = fast.perminverse(perm)
features = features[pinv]
#start = 792320
#end = 792320 + 23*256

R = rosch_analysis(features, hvm_dataset, ntest=5, do_centroids=False, var_levels=['V6'])

