import sys
import cPickle
import math
import numpy as np
import os
import h5py

def generate_list(img_dir, size, ext):
    with open("/om/user/hyo/caffe/angldiff/data/list_" + ext + '.txt', "w") as f:
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
    with open("/om/user/hyo/caffe/angldiff/data/list_" + ext + "_front.txt", "w") as f:
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
    with open("/om/user/hyo/caffe/angldiff/data/list_" + ext + "_pair.txt", "w") as f:
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

def generate_data_3d(meta_path, img_dir, ext, wBack):
    meta = cPickle.load(open(meta_path))
    if 'rxy_sin' not in meta.dtype.fields.keys():
        ryz_cos = np.cos(meta['ryz1']*math.pi/180)
        ryz_sin = np.sin(meta['ryz1']*math.pi/180)
        rxz_cos = np.cos(meta['rxz1']*math.pi/180)
        rxz_sin = np.sin(meta['rxz1']*math.pi/180)
        rxy_cos = np.cos(meta['rxy1']*math.pi/180)
        rxy_sin = np.sin(meta['rxy1']*math.pi/180)
        meta = meta.addcols([ryz_cos, ryz_sin, rxz_cos, rxz_sin, rxy_cos, rxy_sin], names=['ryz_cos', 'ryz_sin', 'rxz_cos', 'rxz_sin', 'rxy_cos', 'rxy_sin'])
        with open(meta_path, "w") as f:
            cPickle.dump(meta, f)
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
    label = np.concatenate( (np.array(meta_p['ryz_sin']).reshape(len(meta),1),
                             np.array(meta_p['ryz_cos']).reshape(len(meta),1),
                             np.array(meta_p['rxy_sin']).reshape(len(meta),1),
                             np.array(meta_p['rxy_cos']).reshape(len(meta),1),
                             np.array(meta_p['rxz_sin']).reshape(len(meta),1),
                             np.array(meta_p['rxz_cos']).reshape(len(meta),1)), axis=1)
    label = label.astype('float32')
    
    with h5py.File('data/train_label_'+ext+'.h5', 'w') as f:
        f['label'] = label[:train_range]
    with h5py.File('data/test_label_'+ext+'.h5', 'w') as f:
        f['label'] = label[train_range:]
    with open('data/train_label_'+ext+'.txt', 'w') as f:
        f.write('/om/user/hyo/caffe/angldiff/data/train_label_'+ext+'.h5\n')
    with open('data/test_label_'+ext+'.txt', 'w') as f:
        f.write('/om/user/hyo/caffe/angldiff/data/test_label_'+ext+'.h5\n')

    # generate labels for pair
    ryz = meta_p['ryz1'] - meta_p['ryz1'][index]
    rxy = meta_p['rxy1'] - meta_p['rxy1'][index]
    rxz = meta_p['rxz1'] - meta_p['rxz1'][index]
    ryz_diff =  (2*((-180<ryz) & (ryz<=180))-1) * np.sign(ryz) * np.minimum(np.abs(ryz), 360-np.abs(ryz))
    rxy_diff =  (2*((-180<rxy) & (rxy<=180))-1) * np.sign(rxy) * np.minimum(np.abs(rxy), 360-np.abs(rxy))
    rxz_diff =  (2*((-180<rxz) & (rxz<=180))-1) * np.sign(rxz) * np.minimum(np.abs(rxz), 360-np.abs(rxz))
    ryz_cos = np.cos(ryz_diff*math.pi/180)
    ryz_sin = np.sin(ryz_diff*math.pi/180)
    rxz_cos = np.cos(rxz_diff*math.pi/180)
    rxz_sin = np.sin(rxz_diff*math.pi/180)
    rxy_cos = np.cos(rxy_diff*math.pi/180)
    rxy_sin = np.sin(rxy_diff*math.pi/180)

    label_pair = np.concatenate( (np.array(ryz_sin).reshape(len(meta),1),
                             np.array(ryz_cos).reshape(len(meta),1),
                             np.array(rxy_sin).reshape(len(meta),1),
                             np.array(rxy_cos).reshape(len(meta),1),
                             np.array(rxz_sin).reshape(len(meta),1),
                             np.array(rxz_cos).reshape(len(meta),1)), axis=1)
    label_pair = label_pair.astype('float32')

    with h5py.File('data/train_label_'+ext+'_pair.h5', 'w') as f:
        f['label'] = label_pair[:train_range]
    with h5py.File('data/test_label_'+ext+'_pair.h5', 'w') as f:
        f['label'] = label_pair[train_range:]
    with open('data/train_label_'+ext+'_pair.txt', 'w') as f:
        f.write('/om/user/hyo/caffe/angldiff/data/train_label_'+ext+'_pair.h5\n')
    with open('data/test_label_'+ext+'_pair.txt', 'w') as f:
        f.write('/om/user/hyo/caffe/angldiff/data/test_label_'+ext+'_pair.h5\n')

def main():
  # ext = "class5"
  # ext_data = "simple5"
  ext_data = sys.argv[1] 
  ext = sys.argv[2]
  wBack = (sys.argv[3] == 'True')
  meta_path = '/om/user/hyo/.skdata/genthor/RoschDataset3_' + ext_data + '_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/meta.pkl'
  img_dir = '/om/user/hyo/.skdata/genthor/RoschDataset3_' + ext_data + '_6eef6648406c333a4035cd5e60d0bf2ecf2606d7/cache/568ce4d00d2c7901515e71c0f90628db084f9dc6/jpeg/'
  generate_data_3d(meta_path, img_dir, ext, wBack)

if __name__ == "__main__":
    main()
