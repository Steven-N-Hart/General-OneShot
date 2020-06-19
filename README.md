# General-OneShot implementation with TF2 Keras


### how to run 
 
```
python train.py \
-m ResNet152 -c 128 \
-o RMSprop \
-p 256 \
-t <path to training images> \
-v <path to validation images> \
-l  <path to log dir>   \
-r 0.00001 \
-e 40 -b 32 -V DEBUG --filetype images

```

### example: cat train.sh
```
python /projects/shart/digital_pathology/scripts/General-OneShot/train.py \
-m ResNet152 -c 128 \
-o RMSprop \
-p 256 \
-t /projects/shart/digital_pathology/data/dogs_vs_cats/train \
-v /projects/shart/digital_pathology/data/dogs_vs_cats/val \
-l /projects/shart/digital_pathology/data/dogs_vs_cats/log_hardsamp_test  \
-r 0.00001 \
-e 40 -b 32 -V DEBUG --filetype images
```
### Examples of training& validation images directory (assume you have two categories(like normal and cancer) in your data i.e 0 and 1)

```
├── train
│   ├── 0
│   │   ├── 1.jpg
│   │   └── 2.jpg
│   └── 1
│       ├── 1.jpg
│       └── 2.jpg
└── validation
    ├── 0
    │   └── 2.jpg
    └── 1
        ├── 1.jpg
        └── 2.jpg
```