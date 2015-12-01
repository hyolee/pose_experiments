import sys
import cPickle
import math
import numpy as np
import os
import h5py
import tabular
import Quaternion as Q
import yamutils.fast as fast

def generate_list(im_pth, ext, train_test):
    size = len(im_pth)
    with open("/om/user/hyo/caffe/quatdiff/data/" + train_test + "_" + ext + '.txt', "w") as f:
        for i in range(size/100):
            buf = ""
            for j in range(5):
                no = i*100 + j*20
                buf = buf + im_pth[no] + " 0\n"
                buf = buf + im_pth[no+1] + " 0\n"
                buf = buf + im_pth[no+2] + " 0\n"
                buf = buf + im_pth[no+3] + " 0\n"
                buf = buf + im_pth[no+4] + " 0\n"
                buf = buf + im_pth[no+5] + " 0\n"
                buf = buf + im_pth[no+6] + " 0\n"
                buf = buf + im_pth[no+7] + " 0\n"
                buf = buf + im_pth[no+8] + " 0\n"
                buf = buf + im_pth[no+9] + " 0\n"
                buf = buf + im_pth[no+10] + " 0\n"
                buf = buf + im_pth[no+11] + " 0\n"
                buf = buf + im_pth[no+12] + " 0\n"
                buf = buf + im_pth[no+13] + " 0\n"
                buf = buf + im_pth[no+14] + " 0\n"
                buf = buf + im_pth[no+15] + " 0\n"
                buf = buf + im_pth[no+16] + " 0\n"
                buf = buf + im_pth[no+17] + " 0\n"
                buf = buf + im_pth[no+18] + " 0\n"
                buf = buf + im_pth[no+19] + " 0\n"
            f.write( buf )

        remainder = size/100 * 100
        for i in range(remainder, size):
            f.write( im_pth[i] + " 0\n")

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
    get_Q = lambda x: euler2q(x[0], x[1], x[2]) 
    label = np.array(map(get_Q, A))    
    return label

def get_quat_diff(q, q0):
    Qq = Q.Quat(q)
    Qq0 = Q.Quat(q0)
    Qdiff = Qq * Qq0.inv()
    return Qdiff.q

def generate_label_pair(A, A_pair):
    Qq = generate_label(A)
    Q_pair = generate_label(A_pair)
    label_pair = np.array([ get_quat_diff(Qq[i,:], Q_pair[i,:]) for i in range(len(Qq)) ])
    # Test
    if ( (np.array([ (Q.Quat(label_pair[i,:]) * Q.Quat(Q_pair[i,:])).q for i in range(len(Qq)) ]) - Qq) < 0.001 ).all():
        print "(Test message) quaternion difference for labels is computed correctly..."
    else:
      return -1
    return label_pair

def get_meta(category):
    return cPickle.load(open('/om/user/hyo/.skdata/genthor/RoschDataset3_' + category + '_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/meta.pkl'))
def get_im_pth(category):
    return '/om/user/hyo/.skdata/genthor/RoschDataset3_' + category + '_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/cache/568ce4d00d2c7901515e71c0f90628db084f9dc6/jpeg/'
def get_im_front_pth(obj, wBack):
    pth = '/om/user/hyo/caffe/angldiff/image_canonical/'
    if wBack:
        pth = pth[:-1] + '_wBack/'
    return pth + obj + '.jpeg'

def generate_index(meta_p, ext):
    size = len(meta_p)
    train_range = size - size/256/11 * 256
    test_range = size - train_range
    index_train = np.arange(0, train_range)
    index_test = np.arange(train_range, size)
    if 'BATCH' in ext:
        objs = list(set(meta_p['obj']))
        v = np.arange(len(meta_p))
        index_train = np.array([], dtype=int)
        index_test = np.array([], dtype=int)
        for obj in objs:
          index_obj = v[meta_p['obj']==obj]
          n_train = len(index_obj) - len(index_obj)/11
          index_train = np.append( index_train, index_obj[:n_train] )
          index_test = np.append( index_test, index_obj[n_train:] )
    return (index_train, index_test)

def generate_data(categories, ext, wBack):
    def get_nCAT(ext,len_meta):
        nCAT = 51200 if 'LESS' in ext else len_meta
        return nCAT
    meta = get_meta(categories[0])
    perm = np.random.RandomState(seed=0).permutation(len(meta))
    meta = meta[perm][:get_nCAT(ext,len(meta))]
    im_pth = [get_im_pth(categories[0]) + str(i) + '.jpeg' for i in range(len(meta))][:get_nCAT(ext,len(meta))]
    if len(categories) > 1:
        for cat in categories[1:]:
            meta_cat = get_meta(cat)
            perm = np.random.RandomState(seed=0).permutation(len(meta_cat))
            meta = tabular.tab_rowstack((meta, meta_cat[perm][:get_nCAT(ext,len(meta))]))
            im_pth = im_pth + [get_im_pth(cat) + str(i) + '.jpeg' for i in range(len(meta_cat))][:get_nCAT(ext,len(meta))]
        perm = np.random.RandomState(seed=0).permutation(len(meta))
        meta_p = meta[perm]
        im_pth = np.array(im_pth)[perm]
    else:
        meta_p = meta
        im_pth = np.array(im_pth)

    # get index
    index_train, index_test = generate_index(meta_p, ext)

    # generate data list
    
    generate_list(im_pth[index_train], ext, 'train')
    generate_list(im_pth[index_test], ext, 'test')

    # generate front data list
    im_front_pth = np.array([get_im_front_pth(obj, wBack) for obj in meta_p['obj']])
    generate_list(im_front_pth[index_train], ext+'_front', 'train')
    generate_list(im_front_pth[index_test], ext+'_front', 'test')

    # generate labels
    A = np.concatenate( (np.array(meta_p['ryz1']).reshape(len(meta),1),
                             np.array(meta_p['rxz1']).reshape(len(meta),1),
                             np.array(meta_p['rxy1']).reshape(len(meta),1)), axis=1)
    A = A.astype('float32')
    label = generate_label(A)
    with h5py.File('data/train_label_'+ext+'.h5', 'w') as f:
        f['label'] = label[index_train]
    with h5py.File('data/test_label_'+ext+'.h5', 'w') as f:
        f['label'] = label[index_test]
    with open('data/train_label_'+ext+'.txt', 'w') as f:
        f.write('/om/user/hyo/caffe/quatdiff/data/train_label_'+ext+'.h5\n')
    with open('data/test_label_'+ext+'.txt', 'w') as f:
        f.write('/om/user/hyo/caffe/quatdiff/data/test_label_'+ext+'.h5\n')

def main():
  # ext = "class5"
  # ext_data = "simple5"
  cat = sys.argv[1]
  categories = cat.split('_')
  ext = "class" + str(len(categories)) + "CAT" + cat
  if len(sys.argv) > 3:
      # BATCH means training batches of objects at a time
      ext = ext + sys.argv[3]
  wBack = (sys.argv[2] == 'True')
  generate_data(categories, ext, wBack)

if __name__ == "__main__":
    main()
