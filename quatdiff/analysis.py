import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import misc
import cPickle
import math
import numpy as np
import Quaternion as Q
import caffe

def get_features(train_val, caffemodel, layer, nBatch, do_save=False, output_file=''):
    net = caffe.Net(train_val, caffemodel, caffe.TEST)

    for iter in range(nBatch):
        out = net.forward()
        if iter == 0:
            features = net.blobs[layer].data
        else:
            features = np.append(features, net.blobs[layer].data, axis=0)
    features = features.reshape(features.shape[0], features.shape[1])

    if (do_save):
      with open("F_" + output_file + ".p", "wb") as f:
          cPickle.dump(features, f)
    return features

# Convert Quaternions to euler angles
def factor_XYZ(r):
    """
    from http://www.geometrictools.com/Documentation/EulerAngles.pdf
    """
    import math
    pi = math.pi
    if (r[0, 2] < 1):
        if (r[0, 2] > -1):
            thY = math.asin(-r[0, 2]) 
            thX = math.atan2(r[1, 2], r[2, 2])
            thZ = math.atan2(r[0, 1], r[0, 0])
        else:
            thY = - pi/2
            thX = -math.atan2(-r[1, 0], r[1, 1])
            thZ = 0
    else:
        thY = pi/2
        thX = math.atan2(-r[1, 0], r[1, 1])
        thZ = 0
    return np.array( [-thX, thY, thZ] ) * 180 / pi
def q2euler(q):
    def q2rot(q):
        return Q.Quat(q).transform        
    get_euler = lambda x: factor_XYZ( q2rot([x[0], x[1], x[2], x[3]]) )
    return np.array(map(get_euler, q))

# Convert euler angles (in degrees, rx->ry->rz) to Quaternions
def euler2q(rx0, ry0, rz0):
    rx = rx0 * (np.pi / 180) / 2
    ry = ry0 * (np.pi / 180) / 2
    rz = rz0 * (np.pi / 180) / 2
    cz = np.cos(rz)
    sz = np.sin(rz)
    cy = np.cos(ry)
    sy = np.sin(ry)
    cx = np.cos(rx)
    sx = np.sin(rx)
    x = sx*cy*cz - cx*sy*sz
    y = cx*sy*cz + sx*cy*sz
    z = cx*cy*sz - sx*sy*cz
    w = cx*cy*cz + sx*sy*sz
    return Q.Quat(Q.normalize([x, y, z, w])).q

def generate_label(A):
    get_Q = lambda x: euler2q(-x[0], x[1], x[2])
    label = np.array(map(get_Q, A))
    return label

# -- Plotting functions ------- #
def data_specs(inds, meta_cat, meta_obj, im_dir):
    ds = {}
    v = np.arange(len(inds))
    cats = list(set(meta_cat))
    for cat in cats:
        ds[cat] = {}
        objs = list(set(meta_obj[v[meta_cat==cat]]))
        for obj in objs:
            ds[cat][obj] = {}
            inds_obj = v[meta_obj==obj]
            ds[cat][obj]['inds'] = inds_obj
            ds[cat][obj]['im_path'] = im_dir + str(inds[inds_obj[0]]) + ".jpeg"
    return ds

def plot_PvA_perObj(p, a, ds, output_file, titles):
    for cat in ds.keys():
        objs = ds[cat].keys()
        fig, ax = plt.subplots( len(objs), a.shape[1]+1, figsize=(28, 5*len(objs)) )
        if len(objs)==1:
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            im = misc.imread(ds[cat][objs[0]]['im_path'])
            ax[0].imshow(im)
            for i in range(a.shape[1]):
                ax[i+1].scatter(a[:, i], p[:, i])
                ax[i+1].set_xlabel('actual')
                ax[i+1].set_ylabel('predicted')
                ax[i+1].set_title(titles[i], fontsize=15)
        else:
            for k in range(len(objs)):
                a_obj = a[ds[cat][objs[k]]['inds']]
                p_obj = p[ds[cat][objs[k]]['inds']]
                ax[k][0].set_xticks([])
                ax[k][0].set_yticks([])
                im = misc.imread(ds[cat][objs[k]]['im_path'])
                ax[k][0].imshow(im)
                for i in range(a.shape[1]):
                    ax[k][i+1].scatter(a_obj[:, i], p_obj[:, i])
                    ax[k][i+1].set_xlabel('actual')
                    ax[k][i+1].set_ylabel('predicted')
                    ax[k][i+1].set_title(titles[i], fontsize=15)
        fig.savefig(output_file + '_perObj_' + cat + '.png')

    
# ---------------------------- #

def main():
    # get input variables
    from optparse import OptionParser
    print "parsing options..."
    parser = OptionParser()
    parser.add_option("-c", "--nClass", dest="nClass")
    parser.add_option("-l", "--lr", dest="lr")
    parser.add_option("-i", "--iter", dest="iter")
    parser.add_option("-e", "--ext_data", dest="ext_data")
    parser.add_option("-f", "--nBatch", dest="nBatch", type=int, default=117)
    (options, args) = parser.parse_args()
    nClass = options.nClass
    lr = options.lr
    iter = options.iter
    ext_data = options.ext_data
    nBatch = options.nBatch

    # retrieve model specs
    layer = 'q_norm'
    ext = 'class' + nClass
    train_val = "/om/user/hyo/caffe/test_quat/train_val/train_val_" + ext + ".prototxt"
    print "...get model from: " + train_val
    caffemodel = "/om/user/hyo/caffe/test_quat/snapshot/" + ext + "_lr" + str(lr) + "_iter_" + iter + ".caffemodel"
    print "...get model state from: " + caffemodel
    # retrieve data specs
    im_dir = "/om/user/hyo/.skdata/genthor/RoschDataset3_" + ext_data + "_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/cache/568ce4d00d2c7901515e71c0f90628db084f9dc6/jpeg/"
    print "...Load image batches from: " + im_dir
    meta_path = "/om/user/hyo/.skdata/genthor/RoschDataset3_" + ext_data + "_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/meta.pkl"
    print "...retrieve meta from: " + meta_path

    # ------- setup ------- #
    caffe.set_device(0)
    caffe.set_mode_gpu()
    # -------------------- #

    # get predicted quaternions
    q_hat = get_features(train_val, caffemodel, layer, nBatch)
    for i in range(len(q_hat)):
        if q_hat[i,3] < 0:
            q_hat[i,:] = - q_hat[i,:]

    # get actual quaternions
    meta = cPickle.load(open(meta_path))
    perm = np.random.RandomState(0).permutation(len(meta))
    meta_p = meta[perm]
    size = len(meta)
    train_range = size - size/256/11 * 256
    A = np.concatenate( (np.array(meta_p['ryz1']).reshape(len(meta),1),
                             np.array(meta_p['rxz1']).reshape(len(meta),1),
                             np.array(meta_p['rxy1']).reshape(len(meta),1)), axis=1)
    A = A.astype('float32')
    inds = np.arange(train_range,train_range+q_hat.shape[0])
    q = generate_label(A)[inds]

    # define output filename to be saved 
    output_file = "/om/user/hyo/caffe/test_quat/features/figures/" + ext + "_lr" + str(lr) + "_iter_" + iter + '_' + str(nBatch)

    # print plots
    ds = data_specs(inds, meta_p[inds]['category'], meta_p[inds]['obj'], im_dir)
    plot_PvA_perObj(q_hat, q, ds, output_file+'_Q', ['x', 'y', 'z', 'w'] )
    plot_PvA_perObj(q2euler(q_hat), q2euler(q), ds, output_file+'_Euler', ['ryz', 'rxz', 'rxy'])

if __name__ == "__main__":
    main()
