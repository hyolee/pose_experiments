import sys
import cPickle
import math
import numpy as np
import os
import h5py
import Quaternion as Q

def generate_list(img_dir, ext, index, train_test):
    size = len(index)
    with open("/om/user/hyo/caffe/quatdiff/data/"+train_test+"_" + ext + '.txt', "w") as f:
        for i in range(size/100):
            buf = ""
            for j in range(5):
                no = i*100 + j*20
                buf = buf + img_dir + str(index[no]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+1]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+2]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+3]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+4]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+5]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+6]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+7]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+8]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+9]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+10]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+11]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+12]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+13]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+14]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+15]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+16]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+17]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+18]) + ".jpeg 0\n"
                buf = buf + img_dir + str(index[no+19]) + ".jpeg 0\n"
            f.write( buf )

        remainder = size/100 * 100
        for i in range(remainder, size):
            f.write( img_dir + str(index[i]) + ".jpeg 0\n")

def generate_front_list(meta_obj, wBack, ext, index, train_test):
    size = len(index)
    if not wBack:
      img_dir = "/om/user/hyo/caffe/angldiff/image_canonical/"
    else:
      img_dir = "/om/user/hyo/caffe/angldiff/image_canonical_wBack/"
    with open("/om/user/hyo/caffe/quatdiff/data/"+train_test+"_" + ext + "_front.txt", "w") as f:
        for i in range(size/100):
            buf = ""
            for j in range(5):
                no = i*100 + j*20
                buf = buf + img_dir + meta_obj[index[no]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+1]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+2]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+3]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+4]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+5]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+6]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+7]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+8]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+9]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+10]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+11]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+12]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+13]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+14]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+15]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+16]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+17]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+18]] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[index[no+19]] + ".jpeg 0\n"
            f.write( buf )

        remainder = size/100 * 100
        for i in range(remainder, size):
            f.write( img_dir + meta_obj[index[i]] + ".jpeg 0\n")

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

def generate_index(meta_p, ext):
    size = len(meta_p)
    train_range = size - size/256/11 * 256
    test_range = size - train_range
    index_train = np.arange(1, train_range)
    index_test = np.arange(train_range, size)
    # one category
    if '1more' in ext or '1CAT' in ext:
        cat = ext.split('CAT')[-1] if 'CAT' in ext else 'bear'
        objs = list(set(meta_p[meta_p['category']==cat]['obj']))
        if cat=='turtle':
            objs = ['lo_poly_animal_TORTOISE', 'lo_poly_animal_TRTL_B', 'lo_poly_animal_TRTL_GRN', 'lo_poly_animal_TRTL_LBK']
        if cat=='radio':
            objs = ['MB27243', 'MB3055', 'MB30213']
        v = np.arange(len(meta_p))
        index_train = np.array([], dtype=int)
        index_test = np.array([], dtype=int)
        for obj in objs:
          index_obj = v[meta_p['obj']==obj]
          n_train = len(index_obj) - len(index_obj)/256/11*256
          index_train = np.append( index_train, index_obj[:n_train] )
          index_test = np.append( index_test, index_obj[n_train:] )
        index_train = np.sort(index_train)
        index_test = np.sort(index_test)
        if '1more' not in ext:
          index_train = index_train[:93184]
          index_test = index_test[:9216]
#        ext = 'class1more'
    elif '5CAT' in ext:
        cats = (ext.split('CAT')[-1]).split('-')
        v = np.arange(len(meta_p))
        index_train = np.array([], dtype=int)
        index_test = np.array([], dtype=int)
        for cat in cats:
            objs = list(set(meta_p[meta_p['category']==cat]['obj']))
            if cat=='turtle':
                objs = ['lo_poly_animal_TORTOISE', 'lo_poly_animal_TRTL_B', 'lo_poly_animal_TRTL_GRN', 'lo_poly_animal_TRTL_LBK']
            if cat=='radio':
                objs = ['MB27243', 'MB3055', 'MB30213']
            for obj in objs:
              index_obj = v[meta_p['obj']==obj]
              n_train = len(index_obj) - len(index_obj)/256/11*256
              index_train = np.append( index_train, index_obj[:93184] )
              index_test = np.append( index_test, index_obj[93184:102400] )
        index_train = np.sort(index_train)
        index_test = np.sort(index_test)
    elif 'perobj' in ext:
        objs = list(set(meta_p['obj']))
        v = np.arange(len(meta_p))
        index_train = np.array([], dtype=int)
        index_test = np.array([], dtype=int)
        for obj in objs:
          index_obj = v[meta_p['obj']==obj]
          n_train = len(index_obj) - len(index_obj)/256/11*256
          index_train = np.append( index_train, index_obj[:n_train] )
          index_test = np.append( index_test, index_obj[n_train:] )
    if 'LO' in ext:
        inds = np.sort(np.append(index_train, index_test))
        inds_train, inds_test = divide_train_test(meta_p[inds])
        index_train = inds[inds_train]
        index_test = inds[inds_test[:len(index_train)/10]]
    return (index_train, index_test)

def generate_data(meta_path, img_dir, ext, wBack):
    meta = cPickle.load(open(meta_path))
    perm = np.random.RandomState(seed=0).permutation(len(meta))
    meta_p = meta[perm]

    # get index
    index_train, index_test = generate_index(meta_p, ext)

    # generate data list
    generate_list(img_dir, ext, index_train, 'train')
    generate_list(img_dir, ext, index_test, 'test')

    # generate front data list
    generate_front_list(meta_p['obj'], wBack, ext, index_train, 'train')
    generate_front_list(meta_p['obj'], wBack, ext, index_test, 'test')

#    # generate pair data list
#    v = np.arange(len(meta))
#    index = np.empty(len(meta), dtype='int64')
#    objs = set(meta['obj'])
#    for obj in objs:
#      v_obj = v[meta_p['obj']==obj]
#      index[v_obj[:-1]] = v_obj[1:]
#      index[v_obj[-1]] = v_obj[0]
#    generate_pair_list(img_dir, ext, index_pair_train, 'train')  
#    generate_pair_list(img_dir, ext, index_pair_test, 'test')  

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

#    # generate labels for pair
#    A_pair = A[index]
#    label_pair = generate_label_pair(A, A_pair)    
#
#    with h5py.File('data/train_label_'+ext+'_pair.h5', 'w') as f:
#        f['label'] = label_pair[index_pair_train]
#    with h5py.File('data/test_label_'+ext+'_pair.h5', 'w') as f:
#        f['label'] = label_pair[index_pair_test]
#    with open('data/train_label_'+ext+'_pair.txt', 'w') as f:
#        f.write('/om/user/hyo/caffe/quatdiff/data/train_label_'+ext+'_pair.h5\n')
#    with open('data/test_label_'+ext+'_pair.txt', 'w') as f:
#        f.write('/om/user/hyo/caffe/quatdiff/data/test_label_'+ext+'_pair.h5\n')

def main():
  # ext = "class5"
  # ext_data = "simple5"
  ext_data = sys.argv[1] 
  ext = sys.argv[2]
  wBack = (sys.argv[3] == 'True')
  meta_path = '/om/user/hyo/.skdata/genthor/RoschDataset3_' + ext_data + '_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/meta.pkl'
  img_dir = '/om/user/hyo/.skdata/genthor/RoschDataset3_' + ext_data + '_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/cache/568ce4d00d2c7901515e71c0f90628db084f9dc6/jpeg/'
  generate_data(meta_path, img_dir, ext, wBack)

if __name__ == "__main__":
    main()
