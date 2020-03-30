#python /projects/shart/digital_pathology/scripts/General-OneShot/train.py -t /projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/sample_images/train -v /projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/sample_images/val -m ResNet50 -o RMSprop -p 256 -l /projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/sample_images/triplet_lossless_images_2 -r 0.001 -e 50 -b 64  -V DEBUG --use-multiprocessing True 
#exit
#--use-multiprocessing True --filetype tfrecords --tfrecord_label 'phenotype/muttype' --tfrecord_image 'image/encoded' --train_num_layers 0
#python train.py -t /projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/TF2.Final_TFRecords/train -v /projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/TF2.Final_TFRecords/val -m ResNet50 -o RMSprop -p 256 -l /projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/sample_images/triplet_lossless -r 0.001 -e 30 -b 64  -V DEBUG --use-multiprocessing True --use-multiprocessing True --filetype tfrecords --tfrecord_label 'phenotype/muttype' --tfrecord_image 'image/encoded' --train_num_layers 0
#python /projects/shart/digital_pathology/scripts/General-Oneshot-Clean/General-OneShot/train.py  \
#-t /projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/TF2.Final_TFRecords/train \
#-v /projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/TF2.Final_TFRecords/val \
#-m ResNet50 \
#-o RMSprop \
#-p 256 \
#-l /projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/sample_images/triplet_lossless_tfrecord1 \
#-r 0.01 \
#-e 50 -b 64  -V DEBUG \
#--use-multiprocessing True --filetype tfrecords --tfrecord_label 'phenotype/subtype' --tfrecord_image 'image/encoded'
# --train_num_layers 0 
#exit
python /projects/shart/digital_pathology/scripts/General-Oneshot-Clean/General-OneShot/train.py  \
-t /projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/sample_images/train  \
-v /projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/sample_images/val \
-m ResNet50 \
-o RMSprop \
-p 256 \
-l /projects/shart/digital_pathology/data/biliary_2020/annotations/images/Images_QC/sample_images/triplet_lossless_tfrecord_img \
-r 0.001 \
-e 1 -b 32  -V DEBUG --filetype images
#--use-multiprocessing True --filetype images
#-L BinaryCrossentropy
