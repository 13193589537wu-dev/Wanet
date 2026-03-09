#### 介绍
deep learning
###

### Data Resources
Due to the ongoing peer-review process and privacy considerations, only a representative subset of the dataset (sample images and annotations) is currently made public in the samples/ directory. The complete North China Plain Ditch Dataset (NCPD) and the pre-trained WANet models will be fully released upon the official publication of the manuscript.
We seek your kind understanding for not providing the full dataset at this stage due to privacy concerns！！！！

### Training
#### 1 gpus
bash tools/dist_train.sh configs/wanet/Wanet.py 1
#### 4 gpus
bash tools/dist_train.sh configs/wanet/Wanet.py 4
### Testing 
python tools/test.py configs/wanet/Wanet.py /path/to/... --eval mIoU

