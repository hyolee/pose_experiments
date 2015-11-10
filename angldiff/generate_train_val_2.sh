#!/bin/sh

nClass=$1

cat << EOF
name: "AlexNet"
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "old_label"
  image_data_param {
    source: "/om/user/hyo/caffe/angldiff/data/train_class${nClass}.txt"
    batch_size: 128
  }
  transform_param {
    crop_size: 227
    mean_file: "/om/user/hyo/caffe/imagenet_mean.binaryproto"
    mirror: false
  }
  include: { phase: TRAIN }
}
layer {
  name: "data_f"
  type: "ImageData"
  top: "data_f"
  top: "old_label_f"
  image_data_param {
    source: "/om/user/hyo/caffe/angldiff/data/train_class${nClass}_front.txt"
    batch_size: 128
  }
  transform_param {
    crop_size: 227
    mean_file: "/om/user/hyo/caffe/imagenet_mean.binaryproto"
    mirror: false
  }
  include: { phase: TRAIN }
}
layer {
  name: "label"
  type: "HDF5Data"
  top: "label"
  hdf5_data_param {
    source: "/om/user/hyo/caffe/angldiff/data/train_label_class${nClass}.txt"
    batch_size: 128
  }
  include: { phase: TRAIN }
}
layer {
  name: "data"
  type: "ImageData"
  top: "data"
  top: "old_label"
  image_data_param {
    source: "/om/user/hyo/caffe/angldiff/data/test_class${nClass}.txt"
    batch_size: 50
  }
  transform_param {
    crop_size: 227
    mean_file: "/om/user/hyo/caffe/imagenet_mean.binaryproto"
    mirror: false
  }
  include: { phase: TEST }
}
layer {
  name: "data_f"
  type: "ImageData"
  top: "data_f"
  top: "old_label_f"
  image_data_param {
    source: "/om/user/hyo/caffe/angldiff/data/test_class${nClass}_front.txt"
    batch_size: 50
  }
  transform_param {
    crop_size: 227
    mean_file: "/om/user/hyo/caffe/imagenet_mean.binaryproto"
    mirror: false
  }
  include: { phase: TEST }
}
layer {
  name: "label"
  type: "HDF5Data"
  top: "label"
  hdf5_data_param {
    source: "/om/user/hyo/caffe/angldiff/data/test_label_class${nClass}.txt"
    batch_size: 50
  }
  include: { phase: TEST }
}
layer {
  name: "silence"
  type: "Silence"
  bottom: "old_label"
}
layer {
  name: "silence_f"
  type: "Silence"
  bottom: "old_label_f"
}

layer {
  name: "conv1"
  type: "Convolution"
  bottom: "data"
  top: "conv1"
  param { 
    name: "conv1_w"
    lr_mult: 1
  }
  param {
    name: "conv1_b"
    lr_mult: 2
  }
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
layer {
  name: "relu1"
  type: "ReLU"
  bottom: "conv1"
  top: "conv1"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv2"
  type: "Convolution"
  bottom: "pool1"
  top: "conv2"
  param {
    name: "conv2_w"
    lr_mult: 1
  }
  param {
    name: "conv2_b"
    lr_mult: 2
  }
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
layer {
  name: "relu2"
  type: "ReLU"
  bottom: "conv2"
  top: "conv2"
}
layer {
  name: "pool2"
  type: "Pooling"
  bottom: "conv2"
  top: "pool2"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv3"
  type: "Convolution"
  bottom: "pool2"
  top: "conv3"
  param {
    name: "conv3_w"
    lr_mult: 1
  }
  param {
    name: "conv3_b"
    lr_mult: 2
  }
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
layer {
  name: "relu3"
  type: "ReLU"
  bottom: "conv3"
  top: "conv3"
}
layer {
  name: "conv4"
  type: "Convolution"
  bottom: "conv3"
  top: "conv4"
  param {
    name: "conv4_w"
    lr_mult: 1
  }
  param {
    name: "conv4_b"
    lr_mult: 2
  }
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
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
layer {
  name: "relu4"
  type: "ReLU"
  bottom: "conv4"
  top: "conv4"
}
layer {
  name: "conv5"
  type: "Convolution"
  bottom: "conv4"
  top: "conv5"
  param {
    name: "conv5_w"
    lr_mult: 1
  }
  param {
    name: "conv5_b"
    lr_mult: 2
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
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
layer {
  name: "relu5"
  type: "ReLU"
  bottom: "conv5"
  top: "conv5"
}
layer {
  name: "pool5"
  type: "Pooling"
  bottom: "conv5"
  top: "pool5"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "fc6"
  type: "InnerProduct"
  bottom: "pool5"
  top: "fc6"
  param {
    name: "fc6_w"
    lr_mult: 1
  }
  param {
    name: "fc6_b"
    lr_mult: 2
  }
  inner_product_param {
    num_output: 4096
    weight_filler {
      type: "xavier"
      #std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu6"
  type: "ReLU"
  bottom: "fc6"
  top: "fc6"
}
layer {
  name: "fc7"
  type: "InnerProduct"
  bottom: "fc6"
  top: "fc7"
  param {
    name: "fc7_w"
    lr_mult: 1
  }
  param {
    name: "fc7_b"
    lr_mult: 2
  }
  inner_product_param {
    num_output: 4096
    weight_filler {
       type: "xavier"
#      type: "xavier"
#      #std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu7"
  type: "ReLU"
  bottom: "fc7"
  top: "fc7"
}
layer {
  name: "fc_pose"
  type: "InnerProduct"
  bottom: "fc7"
  top: "fc_pose"
  param { 
    name: "fc_pose_w"
    lr_mult: 1 
  }
  param { 
    name: "fc_pose_b"
    lr_mult: 2
  }
  inner_product_param {
    num_output: 6
    weight_filler {
       type: "xavier"
#      #std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}

layer {
  name: "conv1_f"
  type: "Convolution"
  bottom: "data_f"
  top: "conv1_f"
  param { 
    name: "conv1_w"
    lr_mult: 1
  }
  param {
    name: "conv1_b"
    lr_mult: 2
  }
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
layer {
  name: "relu1_f"
  type: "ReLU"
  bottom: "conv1_f"
  top: "conv1_f"
}
layer {
  name: "pool1_f"
  type: "Pooling"
  bottom: "conv1_f"
  top: "pool1_f"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv2_f"
  type: "Convolution"
  bottom: "pool1_f"
  top: "conv2_f"
  param {
    name: "conv2_w"
    lr_mult: 1
  }
  param {
    name: "conv2_b"
    lr_mult: 2
  }
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
layer {
  name: "relu2_f"
  type: "ReLU"
  bottom: "conv2_f"
  top: "conv2_f"
}
layer {
  name: "pool2_f"
  type: "Pooling"
  bottom: "conv2_f"
  top: "pool2_f"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "conv3_f"
  type: "Convolution"
  bottom: "pool2_f"
  top: "conv3_f"
  param {
    name: "conv3_w"
    lr_mult: 1
  }
  param {
    name: "conv3_b"
    lr_mult: 2
  }
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
layer {
  name: "relu3_f"
  type: "ReLU"
  bottom: "conv3_f"
  top: "conv3_f"
}
layer {
  name: "conv4_f"
  type: "Convolution"
  bottom: "conv3_f"
  top: "conv4_f"
  param {
    name: "conv4_w"
    lr_mult: 1
  }
  param {
    name: "conv4_b"
    lr_mult: 2
  }
  convolution_param {
    num_output: 384
    pad: 1
    kernel_size: 3
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
layer {
  name: "relu4_f"
  type: "ReLU"
  bottom: "conv4_f"
  top: "conv4_f"
}
layer {
  name: "conv5_f"
  type: "Convolution"
  bottom: "conv4_f"
  top: "conv5_f"
  param {
    name: "conv5_w"
    lr_mult: 1
  }
  param {
    name: "conv5_b"
    lr_mult: 2
  }
  convolution_param {
    num_output: 256
    pad: 1
    kernel_size: 3
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
layer {
  name: "relu5_f"
  type: "ReLU"
  bottom: "conv5_f"
  top: "conv5_f"
}
layer {
  name: "pool5_f"
  type: "Pooling"
  bottom: "conv5_f"
  top: "pool5_f"
  pooling_param {
    pool: MAX
    kernel_size: 3
    stride: 2
  }
}
layer {
  name: "fc6_f"
  type: "InnerProduct"
  bottom: "pool5_f"
  top: "fc6_f"
  param {
    name: "fc6_w"
    lr_mult: 1
  }
  param {
    name: "fc6_b"
    lr_mult: 2
  }
  inner_product_param {
    num_output: 4096
    weight_filler {
      type: "xavier"
      #std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu6_f"
  type: "ReLU"
  bottom: "fc6_f"
  top: "fc6_f"
}
layer {
  name: "fc7_f"
  type: "InnerProduct"
  bottom: "fc6_f"
  top: "fc7_f"
  param {
    name: "fc7_w"
    lr_mult: 1
  }
  param {
    name: "fc7_b"
    lr_mult: 2
  }
  inner_product_param {
    num_output: 4096
    weight_filler {
       type: "xavier"
#      type: "xavier"
#      #std: 0.005
    }
    bias_filler {
      type: "constant"
      value: 0.1
    }
  }
}
layer {
  name: "relu7_f"
  type: "ReLU"
  bottom: "fc7_f"
  top: "fc7_f"
}
layer {
  name: "fc_pose_f"
  type: "InnerProduct"
  bottom: "fc7_f"
  top: "fc_pose_f"
  param { 
    name: "fc_pose_w"
    lr_mult: 1 
  }
  param { 
    name: "fc_pose_b"
    lr_mult: 2
  }
  inner_product_param {
    num_output: 6
    weight_filler {
       type: "xavier"
#      #std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}

layer {
  name: "diff"
  type: "Eltwise"
  bottom: "fc_pose"
  bottom: "fc_pose_f"
  top: "diff"
  eltwise_param {
    operation: SUM
    coeff: 1
    coeff: -1
  }
}
layer {
  name: "loss"
  type: "EuclideanLoss"
  bottom: "diff"
  bottom: "label"
  top: "loss"
}
EOF
