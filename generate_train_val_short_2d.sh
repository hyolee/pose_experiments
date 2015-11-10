#!/bin/sh

nClass=$1

cat << EOF
name: "AlexNet"
layers {
  name: "data"
  #type: HDF5_DATA
  type: IMAGE_DATA
  top: "data"
  top: "old_label"
  image_data_param {
    source: "/om/user/hyo/caffe/reg/train_class${nClass}_2d.txt"
    batch_size: 256
  }
  transform_param {
    crop_size: 227
    mean_file: "imagenet_mean.binaryproto"
    mirror: false
  }
  include: { phase: TRAIN }
}
layers {
  name: "label"
  type: HDF5_DATA
  top: "label"
  top: "label2"
  hdf5_data_param {
    source: "/om/user/hyo/caffe/reg/hdf5_reg_train_label_class${nClass}_2d.txt"
    batch_size: 256
  }
  include: { phase: TRAIN }
}
layers {
  name: "data"
  #type: HDF5_DATA
  type: IMAGE_DATA
  top: "data"
  top: "old_label"
  image_data_param {
    source: "/om/user/hyo/caffe/reg/test_class${nClass}_2d.txt"
    batch_size: 50
  }
  transform_param {
    crop_size: 227
    mean_file: "imagenet_mean.binaryproto"
    mirror: false
  }
  include: { phase: TEST }
}
layers {
  name: "label"
  type: HDF5_DATA
  top: "label"
  top: "label2"
  hdf5_data_param {
    source: "/om/user/hyo/caffe/reg/hdf5_reg_test_label_class${nClass}_2d.txt"
    batch_size: 50
  }
  include: { phase: TEST }
}

layers {
  name: "silence"
  type: SILENCE
  bottom: "old_label"
}
#layers {
#  name: "debug"
#  type: CONVOLUTION
#  bottom: "data"
#  convolution_param {
#    num_output: 1
#    kernel_size: 227
#    weight_filler {
#      type: "constant"
#      value: 1
#    }	     
#  }		    
#  top: "debug"
#}
layers {
  name: "conv1"
  type: CONVOLUTION
  bottom: "data"
  top: "conv1"
  blobs_lr: 1
  blobs_lr: 2
  convolution_param {
    num_output: 96
    kernel_size: 11
    stride: 4
    weight_filler {
      type: "xavier"
      #std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  name: "relu1"
  type: RELU
  bottom: "conv1"
  top: "conv1"
}
layers {
  name: "pool1"
  type: POOLING
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layers {
  name: "conv2"
  type: CONVOLUTION
  bottom: "pool1"
  top: "conv2"
  blobs_lr: 1
  blobs_lr: 2
  convolution_param {
    num_output: 256
    pad: 2
    kernel_size: 5
    group: 2
    weight_filler {
      type: "xavier"
      #std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layers {
  name: "relu2"
  type: RELU
  bottom: "conv2"
  top: "conv2"
}
layers {
  name: "pool2"
  type: POOLING
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layers {
  name: "conv3"
  type: CONVOLUTION
  bottom: "pool2"
  top: "conv3"
  blobs_lr: 1
  blobs_lr: 2
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "xavier"
      #std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  name: "relu3"
  type: RELU
  bottom: "conv3"
  top: "conv3"
}
layers {
  name: "pool3"
  type: POOLING
  bottom: "conv3"
  top: "pool3"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layers {
  name: "fc6"
  type: INNER_PRODUCT
  bottom: "pool3"
  top: "fc6"
  blobs_lr: 1
  blobs_lr: 2
  inner_product_param {
    num_output: 200
    weight_filler {
      type: "xavier"
      #std: 0.005
    }
    bias_filler {
      type: "constant"
    }
  }
}
layers {
  name: "relu6"
  type: RELU
  bottom: "fc6"
  top: "fc6"
}
layers {
  name: "fc7"
  type: INNER_PRODUCT
  bottom: "fc6"
  top: "fc7"
  blobs_lr: 1
  blobs_lr: 2
  inner_product_param {
    num_output: 200
    weight_filler {
       type: "xavier"
#      type: "xavier"
#      #std: 0.005
    }
    bias_filler {
      type: "constant"
    }
  }
}
layers {
  name: "relu7"
  type: RELU
  bottom: "fc7"
  top: "fc7"
}
layers {
  name: "fc_pose"
  type: INNER_PRODUCT
  bottom: "fc7"
  top: "fc_pose"
  blobs_lr: 1
  blobs_lr: 2
  inner_product_param {
    num_output: 2
    weight_filler {
       type: "xavier"
#      type: "xavier"
#      #std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
layers {
  name: "label_vec"
  type: CONCAT
  bottom: "label"
  bottom: "label2"
  top: "label_vec"
}
layers {
  name: "loss"
  type: EUCLIDEAN_LOSS
  bottom: "fc_pose"
  bottom: "label_vec"
  top: "loss"
}
EOF
