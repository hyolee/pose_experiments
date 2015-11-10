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
def q2euler(q):
    x,y,z,w = q
    rx = math.atan2(2*(w*x + y*z), (1 - 2*(x**2 + y**2)))
    ry = math.asin(2*(w*y - x*z))
    rz = math.atan2(2*(w*z + x*y), (1 - 2*(y**2 + z**2)))
    return np.array([rx, ry, rz]) * 180 / np.pi
def convert_equiv(angles):
    def equiv(euler):
        x, y, z = euler
        def mod(x):
            if (x > 180):
                return (x - 360)
            if (x < -180):
                return (x + 360)
            return x
        return [mod(x+180), mod(180-y), mod(z-180)]
    for i in range(len(angles)):
        if (angles[i][0]<-90) or (angles[i][0]>90):
            angles[i] = equiv(angles[i])
    return angles
def convert_q2euler(q):
    get_euler = lambda x: q2euler(x)
    return np.array(convert_equiv(map(get_euler, q)))

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

# For leave-out training
def divide_train_test(meta_p):
    # divide train/test by objs
    v = np.arange(len(meta_p))
    index_train = []
    index_test = []
    import tabular as tb
    m1 = tb.tabarray(columns=[meta_p['obj'], meta_p['category']], names=['obj', 'category'])
    cat_obj_list = m1.aggregate(On=['category'], AggFunc=lambda x: set(x))
    for cat_list in cat_obj_list:
        n_objs = len(cat_list[1])
        n_train = n_objs - (n_objs / 10)
        if n_train == n_objs:
          n_train = n_objs - 1
        obj_list = list(cat_list[1])
        for obj in obj_list[:n_train]:
            index_train = index_train + list(v[meta_p['obj']==obj])
        for obj in obj_list[n_train:]:
            index_test = index_test + list(v[meta_p['obj']==obj])
    index_train.sort()
    index_test.sort()

    return (index_train, index_test)

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
            if 'Q' in output_file:
                ax[0].set_title('Mean error: ' + str(np.sum(np.arccos(abs(np.sum(p*a, axis=1))))/p.shape[0]))
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
                if 'Q' in output_file:
                    ax[k][0].set_title('Mean error: ' + str(np.sum(np.arccos(abs(np.sum(p_obj*a_obj, axis=1))))/p_obj.shape[0]))
                for i in range(a.shape[1]):
                    ax[k][i+1].scatter(a_obj[:, i], p_obj[:, i])
                    ax[k][i+1].set_xlabel('actual')
                    ax[k][i+1].set_ylabel('predicted')
                    ax[k][i+1].set_title(titles[i], fontsize=15)
        fig.savefig(output_file + '_perObj_' + cat + '.png')

def plot_distQ(p, a, output_file):
    e = np.arccos( abs(np.sum(p*a, axis=1)) )
    fig = plt.figure()
    plt.hist(e, bins=np.arange(0, 1.6, 0.1))
    plt.title("Error Distribution - Quaternion Dot-product Distance")
    plt.xlabel('Error (Quaternion Distance)')
    plt.ylabel('Counts') 
    fig.savefig(output_file + 'distQ' + '.png')
    
# ---------------------------- #

def main():
    # get input variables
    from optparse import OptionParser
    print "parsing options..."
    parser = OptionParser()
    parser.add_option("-m", "--model", dest="model", default='1')
    parser.add_option("-c", "--nClass", dest="nClass")
    parser.add_option("-l", "--lr", dest="lr")
    parser.add_option("-i", "--iter", dest="iter")
    parser.add_option("-e", "--ext_data", dest="ext_data")
    parser.add_option("-f", "--nBatch", dest="nBatch", type=int, default=117)
    (options, args) = parser.parse_args()
    model = options.model
    nClass = options.nClass
    lr = options.lr
    iter = options.iter
    ext_data = options.ext_data
    nBatch = options.nBatch

    # retrieve model specs
    layer = 'q_norm'
    ext = model + '_class' + nClass
    train_val = "/om/user/hyo/caffe/quatdiff/train_val/train_val_" + ext + ".prototxt"
    print "...get model from: " + train_val
    caffemodel = "/om/user/hyo/caffe/quatdiff/snapshot/" + ext + "_lr" + str(lr) + "_iter_" + iter + ".caffemodel"
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
    if '1more' in nClass:
        cat = 'bear'
        objs = list(set(meta[meta['category']==cat]['obj']))
        v = np.arange(len(meta))
        inds = np.array([], dtype=int)
        for obj in objs:
          index_obj = v[meta_p['obj']==obj]
          n_train = len(index_obj) - len(index_obj)/256/11*256
          inds = np.append( inds, index_obj[n_train:] )
        inds = np.sort(inds)[:q_hat.shape[0]]
        inds_all = v[meta_p['category']==cat]
    else:
        inds = np.arange(train_range,train_range+q_hat.shape[0])
        inds_all = np.arange(len(meta))
    if 'LO' in nClass:
        inds_train, inds_test = divide_train_test(meta_p[inds_all])
        inds = inds_all[inds_test[:q_hat.shape[0]]] 
    q = generate_label(A[inds])

    # define output filename to be saved 
    output_file = "/om/user/hyo/caffe/quatdiff/features/figures/" + ext + "_lr" + str(lr) + "_iter_" + iter + '_' + str(nBatch)

    # print plots
    ds = data_specs(inds, meta_p[inds]['category'], meta_p[inds]['obj'], im_dir)
    plot_PvA_perObj(q_hat, q, ds, output_file+'_Q', ['x', 'y', 'z', 'w'] )
    plot_PvA_perObj(convert_q2euler(q_hat), convert_q2euler(q), ds, output_file+'_Euler', ['ryz', 'rxz', 'rxy'])
    plot_distQ(q_hat, q, output_file)

if __name__ == "__main__":
    main()
