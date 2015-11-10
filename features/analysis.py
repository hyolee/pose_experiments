import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import misc
import cPickle
import math
import numpy as np
import Quaternion as Q
import caffe
import os
caffe_root = "/om/user/hyo/src/caffe/"
import sys
sys.path.insert(0, caffe_root + 'python')

def get_features(inds, layer, train_val, caffemodel, im_dir):
    net = caffe.Net(train_val, caffemodel, caffe.TEST)
    
    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    #transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    
    # set net to batch size of 50
    #net.blobs['data'].reshape(50,3,227,227)
    # ------------------------#

    no = 0
    nBatch = int(math.ceil(len(inds)/50)) + 1
    print "...Extract features"
    while no < len(inds):
        _in = np.empty(net.blobs['data'].data.shape) 
        for i in range(50):
            im_file = im_dir + str(inds[no]) + ".jpeg"
            _in[i] = transformer.preprocess('data', caffe.io.load_image(im_file))
            no = no + 1
            if no == len(inds):
                break
        net.blobs['data'].data[...] = _in
        out = net.forward()

        if no < 51:
            features = net.blobs[layer].data
        else:
            features = np.append(features, net.blobs[layer].data, axis=0)
    F = features[:len(inds)]
    return F.reshape(F.shape[0], F.shape[1])

def get_error(F, a):
    e = np.sum( (a[:,:6]-F[:,:6])**2 , axis=1) / 3
    return e

def retrieve_angle(sin, cos):
    sin = sin / np.sqrt(sin * sin + cos * cos)
    cos = cos / np.sqrt(sin * sin + cos * cos)
    tmask = (cos > 0)
    theta = (2*tmask - 1) * np.arcsin(sin) + math.pi*(~tmask)
    theta = theta / math.pi * 180
    theta[np.all([(180 < theta), (theta <= 270)], axis=0)] = theta[np.all([(180 < theta), (theta < 270)], axis=0)] - 360
    return theta.reshape(len(theta), 1)

def get_angle_predict(F):
    angle_p = np.concatenate((retrieve_angle(F[:,0], F[:,1]), retrieve_angle(F[:,4], F[:,5]), retrieve_angle(F[:,2], F[:,3])), axis=1)
    return angle_p

def analysis_indiv(F, a, inds, im_dir, output_file):
    print "...print few examples"
    E = get_error(F, a)
    v = np.arange(len(E))
    Is_all = [list(v[E<0.1][:50])]
    Is_all.append(list(v[(0.1<E)&(E<0.5)][:50]))
    Is_all.append(list(v[0.5<E][:50]))
    
    exts = ['good', 'soso', 'bad']
    for j in range(len(Is_all)):
        Is = Is_all[j]
        fig, ax = plt.subplots(len(Is), 2, figsize=(10,5*len(Is)))
        for i in range(len(Is)):
            # draw image
            im = misc.imread(im_dir + str(inds[Is[i]]) + ".jpeg")
            ax[i][0].imshow(im)
            ax[i][0].axis('off')
            
            # draw table
            ax[i][1].set_title('sqsumE: {0:.2f}'.format(E[Is[i]]))
            ax[i][1].axis('off')
            
            cell_text = []
            rows = ['ryz_sin', 'ryz_cos', 'rxy_sin', 'rxy_cos', 'rxz_sin', 'rxz_cos', 'ryz', 'rxz', 'rxy']  #['label' + str(j) for j in range(1,7)]
            columns = ('predict', 'actual')
            for row in range(len(rows)):
                cell_text.append( ["{0:.2f}".format(F[Is[i]][row]), "{0:.2f}".format(a[Is[i]][row])] )
            table = ax[i][1].table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='center')
                
        plt.show()
        fig.savefig(output_file + '_' + exts[j] + '.png')

def analysis_distr(F, a, inds_img, meta_cat, meta_obj, im_dir, output_file):
    print "...print error distribution"
    v = np.arange(len(meta_obj))
    diff = np.abs(F[:,-3:] - a[:, -3:])
    # --- per category
    cats = list(set(meta_cat))
    for cat in cats:
        #inds = v[meta_cat==cat]        
        # --- per object
        # mean error per object
        #meta_here = meta_obj[v]
#        diff = np.abs(F[:,-3:] - a[:, -3:])
        objs = list(set(meta_obj[v[meta_cat==cat]]))
        error_per_obj = []
        for obj in objs:
            e = np.mean(np.amin([diff[meta_obj==obj], 360-diff[meta_obj==obj]], axis=0), axis=0)
            error_per_obj.append(e)

        # percent bad predictions per object
        percent_bad_per_obj = []
    #    v = np.arange(len(meta_obj))
        for k in range(len(objs)):
            obj = objs[k]
            inds = v[meta_obj == obj]
            
            do = diff[inds]
            cou = [0, 0, 0]
            for j in range(len(do)):
                if np.amin([do[j,0], 360-do[j,0]]) > 45:
                    cou[0] = cou[0] + 1
                if np.amin([do[j,1], 360-do[j,1]]) > 45:
                    cou[1] = cou[1] + 1
                if np.amin([do[j,2], 360-do[j,2]]) > 45:
                    cou[2] = cou[2] + 1
            
            percent_bad_per_obj.append( np.array(cou) *100.0 / len(do) )
    
        fig = plt.figure(figsize=(20,5*len(objs)))
        for k in range(len(objs)):
            plt.subplot(len(objs),2, k*2 + 1)
            plt.axis('off')
            ind = inds_img[v[meta_obj == objs[k]][0]]
            im = misc.imread(im_dir + str(ind) + ".jpeg")
            plt.imshow(im)
    
            plt.subplot(len(objs),2, k*2 + 2)
            plt.axis('off')
            plt.title('object: ' + objs[k])
            cell_text = []
            rows = ['mean error', 'percentage > 45']
            columns = ('ryz', 'rxz', 'rxy')
            cell_text.append( ["{0:.2f}".format(error_per_obj[k][0]),"{0:.2f}".format(error_per_obj[k][1]), "{0:.2f}".format(error_per_obj[k][2])] )
            cell_text.append( ["{0:.2f}".format(percent_bad_per_obj[k][0]), "{0:.2f}".format(percent_bad_per_obj[k][1]), "{0:.2f}".format(percent_bad_per_obj[k][2])] )
            table = plt.table(cellText=cell_text, rowLabels=rows, colLabels=columns, loc='center')
    
        plt.show()
        fig.savefig(output_file + '_perObj_' + cat + '.png') 


    # --- per angle
    # predicted vs actual angles
    titles = ['ryz', 'rxz', 'rxy']
    fig, ax = plt.subplots( 3,1,figsize=(10,30) )
    for i in range(3):
        ax[i].scatter(a[:, -3+i], F[:, -3+i])
        ax[i].set_xlabel('actual angle (degrees)')
        ax[i].set_ylabel('predicted angle (degrees)')
        ax[i].set_title(titles[i])
    fig.savefig(output_file + '_perAngl.png')

    # --- per angle, per obj
    for cat in cats:
        objs = list(set(meta_obj[v[meta_cat==cat]])) 
        fig, ax = plt.subplots( len(objs), 4, figsize=(25, 5*len(objs)) )
        for k in range(len(objs)):
            a_obj = a[meta_obj == objs[k]]
            F_obj = F[meta_obj == objs[k]]
            ax[k][0].set_xticks([])
            ax[k][0].set_yticks([])
            im = misc.imread(im_dir + str(inds_img[v[meta_obj == objs[k]][0]]) + ".jpeg")
            ax[k][0].imshow(im) 
            for i in range(3):
                ax[k][i+1].scatter(a_obj[:, -3+i], F_obj[:, -3+i])
                ax[k][i+1].set_xlabel('actual angle (degrees)')
                ax[k][i+1].set_ylabel('predicted angle (degrees)')
                ax[k][i+1].set_title(titles[i])
        fig.savefig(output_file + '_perAnglperObj_' + cat + '.png')

    return 0

def analysis_distr_by_bg(F, a, inds_img, meta_obj, meta_bg, im_dir, output_file):
    print "...print error distribution by background"
    v = np.arange(len(meta_obj))
    diff = np.abs(F[:,-3:] - a[:, -3:])
   
    titles = ['ryz', 'rxz', 'rxy'] 
    objs = list(set(meta_obj))
    for obj in objs:
        bgs = list(set(meta_bg[v[meta_obj==obj]]))
        k = 0
        for k1 in range(len(bgs)/50):
            fig, ax = plt.subplots( 50, 4, figsize=(25, 5*50) )
            k2 = 0
            while k < min((k1+1)*50, len(bgs)):
                a_bg = a[(meta_obj==obj) & (meta_bg == bgs[k])]
                F_bg = F[(meta_obj==obj) & (meta_bg == bgs[k])]
                ax[k2][0].set_xticks([])
                ax[k2][0].set_yticks([])
                im = misc.imread(im_dir + str(inds_img[v[(meta_obj==obj) & (meta_bg == bgs[k])][0]]) + ".jpeg")
                ax[k2][0].imshow(im)
                for i in range(3):
                    ax[k2][i+1].scatter(a_bg[:, -3+i], F_bg[:, -3+i])
                    ax[k2][i+1].set_xlabel('actual angle (degrees)')
                    ax[k2][i+1].set_ylabel('predicted angle (degrees)')
                    ax[k2][i+1].set_title(titles[i])
                k2 = k2 + 1
                k = k1 * 50 + k2
            fig.savefig(output_file + '_perAnglperBgperObj_' + obj + str(k1) + '.png')

# -------- Quaternions
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

def plot_error(pred, actual, output_file):
    e = np.arccos( abs(np.sum(pred*actual, axis=1)) )

    fig = plt.figure()
    plt.hist(e, bins=np.arange(0, 1.6, 0.1))
    plt.title("Error Distribution - Quaternion Dot-product Distance")
    plt.xlabel("Error (Quaternion distance)")
    plt.ylabel("Counts")
    plt.show()
    fig.savefig(output_file + '_QErrorDist.png')

def plot_pred_vs_actual(pred, actual, output_file):
    def m3p(q1, q2):
        return np.arccos(quat_dot(q1, q2))
    def m3m(q1, q2):
        return np.arccos(- quat_dot(q1, q2))

    ep = np.array([ m3p(pred[i], actual[i]) for i in range(len(pred)) ])
    em = np.array([ m3m(pred[i], actual[i]) for i in range(len(pred)) ])
    min_error = (ep < em)

    titles = ['x','y','z','w']
    fig, ax = plt.subplots( 2, 4, figsize=(20, 8) )
    cmap = matplotlib.cm.get_cmap('Blues')
    for r in range(4):
        ax[0,r].scatter( [actual[i].q[r] for i in range(len(actual))],
                       [(2*min_error[i]-1)*pred[i].q[r] for i in range(len(pred))] )
        ax[0,r].set_xlabel('actual')
        ax[0,r].set_ylabel('predicted')
        ax[0,r].set_title('(' + titles[r] + ')')
        ax[0,r].set_xlim(xmin=-1.5, xmax=1.5)
        ax[0,r].set_ylim(ymin=-1.5, ymax=1.5)

        H = ax[1,r].hist2d( [actual[i].q[r] for i in range(len(actual))],
                       [(2*min_error[i]-1)*pred[i].q[r] for i in range(len(pred))],
                       bins = 20, cmap=cmap )
        ax[1,r].set_xlabel('actual')
        ax[1,r].set_ylabel('predicted')
        ax[1,r].set_title('(' + titles[r] + ')')
        ax[1,r].set_xlim(xmin=-1.5, xmax=1.5)
        ax[1,r].set_ylim(ymin=-1.5, ymax=1.5)
    #fig.colorbar( H[3], ax=ax[3] )
    plt.show()
    fig.savefig(output_file + '_Q.png')

def analysis_quat(F, a, inds, meta, im_dir, output_file):
    # get quaternions
    get_Q = lambda x: euler2q(-x[0], x[1], x[2])
    pred = np.array(map(get_Q, F[:,-3:]))
    actual = np.array(map(get_Q, a[:,-3:]))

    # error distribution
    plot_error(pred, actual, output_file)

    # predicted vs actual
    plot_pred_vs_actual(pred, actual, output_file)
#----------

#######

def main(): 
    # get input variables
    from optparse import OptionParser
    print "parsing options..."
    parser = OptionParser()
    parser.add_option("-n", "--net", dest="net")
    parser.add_option("-c", "--nClass", dest="nClass")
    parser.add_option("-d", "--nDim", dest="nDim")
    parser.add_option("-l", "--lr", dest="lr")
    parser.add_option("-i", "--iter", dest="iter")
    parser.add_option("-e", "--ext_data", dest="ext_data")
    parser.add_option("-f", "--nFeatures", dest="nFeatures", type=int, default=5888)
    (options, args) = parser.parse_args()
    net = options.net
    nClass = options.nClass
    nDim = options.nDim
    lr = options.lr
    iter = options.iter
    ext_data = options.ext_data
    nFeatures = options.nFeatures
    
    layer = 'fc_pose'
    ext = 'reg_' + net + '_class' + nClass + '_' + nDim + 'd'
    train_val = "/om/user/hyo/caffe/train_val/train_val_" + ext + ".prototxt"
    print "...get model from: " + train_val
    caffemodel = "/om/user/hyo/caffe/snapshot/caffe_rotation" + nDim + "d_" + ext + "_lr" + str(lr) + "_iter_" + iter + ".caffemodel"
    print "...get model state from: " + caffemodel
    im_dir = "/om/user/hyo/.skdata/genthor/RoschDataset3_" + ext_data + "_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/cache/568ce4d00d2c7901515e71c0f90628db084f9dc6/jpeg/"
    print "...Load image batches from: " + im_dir 

    # ------- setup ------- #
    mode = 'gpu'
    if mode == 'cpu':
        caffe.set_mode_cpu()
    else:
        caffe.set_device(0)
        caffe.set_mode_gpu()
    # -------------------- #
 
    meta_path = "/om/user/hyo/.skdata/genthor/RoschDataset3_" + ext_data + "_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/meta.pkl"
    print "...retrieve meta from: " + meta_path
    meta = cPickle.load(open(meta_path))
    size = len(meta)
    train_range = size - size/256/11 * 256
    
    inds = range(train_range, min(train_range + nFeatures, size))
    
    # get prediction for angles
    F = get_features(inds, layer, train_val, caffemodel, im_dir)
    F = np.append(F, get_angle_predict(F), axis=1)
    
    # get real value for angles
    perm = np.random.RandomState(0).permutation(len(meta))
    meta_p = meta[perm]
    value = ['ryz_sin', 'ryz_cos', 'rxy_sin', 'rxy_cos', 'rxz_sin', 'rxz_cos', 'ryz1', 'rxz1', 'rxy1']
    a = []
    for i in inds:
        vector = [meta_p[i][v] for v in value]
        a.append(vector)
    a = np.array(a)
   
    output_file = "/om/user/hyo/caffe/features/figures/" + ext + "_lr" + str(lr) + "_iter_" + iter + '_rosch' + ext_data + '_' + layer + '_' + str(nFeatures)
    analysis_distr(F, a, inds, meta_p[inds]['category'], meta_p[inds]['obj'], im_dir, output_file) 
    analysis_indiv(F, a, inds, im_dir, output_file)
    #analysis_distr_by_bg(F, a, inds, meta_p[inds]['obj'], meta_p[inds]['bgname'], im_dir, output_file)

    # Quaternion analysis
    analysis_quat(F, a, inds, meta, im_dir, output_file)

if __name__ == "__main__":
    main()
