import sys
import cPickle
import math
import numpy as np
import os
import h5py
import Quaternion as Q

def generate_list(img_dir, size, ext):
    with open("/om/user/hyo/caffe/quatdiff/data/list_" + ext + '.txt', "w") as f:
        for i in range(size/100):
            buf = ""
            for j in range(5):
                no = i*100 + j*20
                buf = buf + img_dir + str(no) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+1) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+2) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+3) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+4) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+5) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+6) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+7) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+8) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+9) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+10) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+11) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+12) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+13) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+14) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+15) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+16) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+17) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+18) + ".jpeg 0\n"
                buf = buf + img_dir + str(no+19) + ".jpeg 0\n"
            f.write( buf )

        remainder = size/100 * 100
        for i in range(remainder, size):
            f.write( img_dir + str(i) + ".jpeg 0\n")

def generate_front_list(meta_obj, wBack, ext, size):
    if not wBack:
      img_dir = "/om/user/hyo/caffe/angldiff/image_canonical/"
    else:
      img_dir = "/om/user/hyo/caffe/angldiff/image_canonical_wBack/"
    with open("/om/user/hyo/caffe/quatdiff/data/list_" + ext + "_front.txt", "w") as f:
        for i in range(size/100):
            buf = ""
            for j in range(5):
                no = i*100 + j*20
                buf = buf + img_dir + meta_obj[no] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+1] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+2] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+3] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+4] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+5] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+6] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+7] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+8] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+9] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+10] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+11] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+12] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+13] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+14] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+15] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+16] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+17] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+18] + ".jpeg 0\n"
                buf = buf + img_dir + meta_obj[no+19] + ".jpeg 0\n"
            f.write( buf )

        remainder = size/100 * 100
        for i in range(remainder, size):
            f.write( img_dir + meta_obj[i] + ".jpeg 0\n")

def generate_pair_list(img_dir, size, ext, index):
    with open("/om/user/hyo/caffe/quatdiff/data/list_" + ext + "_pair.txt", "w") as f:
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
            f.write( img_dir + str(i) + ".jpeg 0\n")

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

def generate_data(meta_path, img_dir, ext, wBack):
    meta = cPickle.load(open(meta_path))
    perm = np.random.RandomState(seed=0).permutation(len(meta))
    meta_p = meta[perm]

    size = len(meta)
    train_range = size - size/256/11 * 256
    test_range = size - train_range

    # generate data list
    generate_list(img_dir, size, ext)
    os.system("head data/list_"+ext+".txt -n "+str(train_range)+" > data/train_"+ext+".txt")
    os.system("tail data/list_"+ext+".txt -n "+str(test_range)+" > data/test_"+ext+".txt")

    # generate front data list
    generate_front_list(meta_p['obj'], wBack, ext, size)
    os.system("head data/list_"+ext+"_front.txt -n "+str(train_range)+" > data/train_"+ext+"_front.txt")
    os.system("tail data/list_"+ext+"_front.txt -n "+str(test_range)+" > data/test_"+ext+"_front.txt")

    # generate pair data list
    v = np.arange(len(meta))
    index = np.empty(len(meta), dtype='int64')
    objs = set(meta['obj'])
    for obj in objs:
      v_obj = v[meta_p['obj']==obj]
      index[v_obj[:-1]] = v_obj[1:]
      index[v_obj[-1]] = v_obj[0]
    generate_pair_list(img_dir, size, ext, index)  
    os.system("head data/list_"+ext+"_pair.txt -n "+str(train_range)+" > data/train_"+ext+"_pair.txt")
    os.system("tail data/list_"+ext+"_pair.txt -n "+str(test_range)+" > data/test_"+ext+"_pair.txt")

    # generate labels
    A = np.concatenate( (np.array(meta_p['ryz1']).reshape(len(meta),1),
                             np.array(meta_p['rxz1']).reshape(len(meta),1),
                             np.array(meta_p['rxy1']).reshape(len(meta),1)), axis=1)
    A = A.astype('float32')
    label = generate_label(A)
    with h5py.File('data/train_label_'+ext+'.h5', 'w') as f:
        f['label'] = label[:train_range]
    with h5py.File('data/test_label_'+ext+'.h5', 'w') as f:
        f['label'] = label[train_range:]
    with open('data/train_label_'+ext+'.txt', 'w') as f:
        f.write('/om/user/hyo/caffe/quatdiff/data/train_label_'+ext+'.h5\n')
    with open('data/test_label_'+ext+'.txt', 'w') as f:
        f.write('/om/user/hyo/caffe/quatdiff/data/test_label_'+ext+'.h5\n')

    # generate labels for pair
    A_pair = A[index]
    label_pair = generate_label_pair(A, A_pair)    

    with h5py.File('data/train_label_'+ext+'_pair.h5', 'w') as f:
        f['label'] = label_pair[:train_range]
    with h5py.File('data/test_label_'+ext+'_pair.h5', 'w') as f:
        f['label'] = label_pair[train_range:]
    with open('data/train_label_'+ext+'_pair.txt', 'w') as f:
        f.write('/om/user/hyo/caffe/quatdiff/data/train_label_'+ext+'_pair.h5\n')
    with open('data/test_label_'+ext+'_pair.txt', 'w') as f:
        f.write('/om/user/hyo/caffe/quatdiff/data/test_label_'+ext+'_pair.h5\n')

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
