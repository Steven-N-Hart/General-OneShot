python /projects/shart/digital_pathology/scripts/General-OneShot/train.py \
-m ResNet152 -c 128 \
-o RMSprop \
-p 256 \
-t /projects/shart/digital_pathology/data/dogs_vs_cats/train \
-v /projects/shart/digital_pathology/data/dogs_vs_cats/val \
-l /projects/shart/digital_pathology/data/dogs_vs_cats/log_hardsamp_test  \
-r 0.00001 \
-e 40 -b 32 -V DEBUG --filetype images
