---
author: "Cong Sun"
title: "使用google Object Detection API检测重力波"
date: 2020-05-21T21:53:28+08:00
lastmod: 2020-05-21T21:53:28+08:00
draft: false
description: ""

tags: ["重力波检测"]
categories: ["机器学习"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: false
auto_collapse_toc: true
math: false
---

在之前的文章中，我已经介绍了tensorflow框架的安装和virtual env下使用的方法。下面这篇博文即是我的毕业设计的项目，同时也是我在学习了机器学习和tensorflow框架一段时间后做的第一个实战。所以大家可以放心的是这篇文章将是平易近人的，不会有太多算法上的介绍（关于这里我使用的网络结构我将在网络结构和算法分析的tag里给大家做详细的介绍）。

<!--more-->

# 项目的简单介绍
大家如果看过我blog的about，应该知道我是一名学习物理相关的学生。我的毕业设计主要是从卫星图像中自动检测重力波。这里读者并不需要知道重力波的具体含义，只需要有一个大体的概念，在卫星图像中大气重力波大概是这样的：
![图像中的重力波](/img/object_detection/1.png)
这样的重力波结构解释我们需要在图像中自动识别出来的目标，下面的内容就是如何使用object detection将它识别出来。
# tensorflow object detection API的简单介绍
其实熟悉tensorflow的人应该都知道，tensorflow在github上的主页是：[tensorflow](https://github.com/tensorflow)，这个主页下面有两个比较重要的repo（可以参照star的数量），分别是Tensorflow的源代码repo：tensorflow/tensorflow，另一个就是我们今天要介绍的tensorflow/models。后者tensorflow/models是Google官方用TensorFlow做的各种实例，对于我们今天这个实践题目比较重要的就是图像分类的Slim。尤其是对于我们现在这个任务，前面必须有一个像样的ImageNet图像分类模型来充当所谓的特征提取（feature extraction）层，比如VGG6、ResNet等网络结构。tensorflow官方实现这些网络结构的项目是tensorflow Slim，而object detection API正是基于slim的，这个库的公布时间比较早，内容也相对丰富和成熟。下面我就从安装到训练再到预测给大家做详细的说明讲解。
# tensorflow/models的安装
首先我们可以参考github上models repo的安装建议：
首先是库依赖情况：
- Protobuf 3+
- Python-tk （这个库在后面做图的时候可能会与Matplotlib由backend的冲突所以要留意一下）
- Pillow 1.0
- lxml（为了识别我们数据集使用的pascal VOC格式数据，一定要安装这个依赖库） 
- Slim（这个库已经包含在tenorflow/models/ersearch/里面，只需要一个指令就可以指定Slim抽取了）
- jupyter notebook（方便的可视化python编程工具，提供良好的交互界面，在编程环境目录的博文中我还会介绍如何使用anaconda3+jupyter实现远程操作）
- Matplotlib（画图库，不解释）
- tensorflow
- cython
- cocoapi

如果您的机器上还没有安装tensoflow的话，请参考我之前的博文进行tensorflow的安装。下面是如何安装上面这些依赖库的介绍（本机环境ubuntu16.04+python3.5.2 使用virtual env）：
```
sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
sudo pip install Cpython
sudo pip install jupyter
sudo pip install matplotlib
```

当然，如果您的机器不是ubuntu或者debian这样使用apt作包管理的系统，那么也可以选择直接用pip安装（在大陆下载速度慢的情况可以通过加-i flag使用清华镜像解决）：
```
sudo pip install Cython
sudo pip install pillow
sudo pip install lxml
sudo pip install jupyter
sudo pip install matplotlib
```

这里由于我的模型没有用到COCO API所以没有安装COCO库，大家如果感兴趣的话可以直接去github上clone COCO下来玩。
在完成了上面的依赖库安装之后，我们直接将github上的models库clone下来，cd目录到tensorflow/models/research 并在目录下执行python安装：
```
python setup.py build
python setup.py install
```

另外需要在tensorflow/models/research中编译protobuf libraries，这是因为我们使用的object detection API要使用protobuf来编译模型并训练参数。如果您的机器中没有安装protobuf，请先编译安装，如果安装后有报错(关于报错的原因我还不是很清楚，如果您知道的话，请不吝赐教)：
```
object_detection/protos/anchor_generator.proto:11:3:  Expected “required”, “optional”, or “repeated”.
object_detection/protos/anchor_generator.proto:11:32: Missing field number. 
```

解决方式是下载protobuf编译后的文件[下载地址](https://www.witsrc.com/download)，解压压缩包到tensorfolw的protoc_3.3文件夹后进入到解压缩文件的bin文件夹
```
cd bin/
pwd(这一步是为了记住你的bin文件夹全地址)
cd ...你的tenforflow地址.../models/
/home/gabrielsun/tensorflow/protoc3.3/bin/protoc objecr_detection/protos/*.proto --python_out=.
```

这样一般就可以成功在research文件夹中成功编译protobuf了。如果还有问题可以在我评论区下留言。
下面我们需要指定使用slim做目标识别，这里值得注意的是如果您不将下面这行代码写入到~/.bashrc文件中的话，那么每新建一个terminal都需要在research文件夹下执行下面的代码：
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim 
```

最后测试是否安装编译好：
```
 python object_detection/builders/models_builder_test.py
```

如果输出有OK即测试通过。
这样我们的安装部分已经结束了！主要的编程环境也已经搭建好了，下面我们讨论数据集构建的问题。
# 数据集的制作

这个部分实际上我我在之前毕业设计的开题中没有好好考虑的部分，可能这也是导致我后来在这个部分花了很大的时间精力。这里简单的说明一下。我的想法是利用图像数据做目标检测，所以如何标记图像是一个很大的问题。我采用的标记方式与imagenet上图像数据集标记方式相同—pascal VOC格式。这个格式主要包括了：
- Images文件夹 （包含训练图片和测试图片，混放在一起）
- Annatations文件夹 （xml格式的标签文件）
- Images set文件夹（在这次的项目中没有用到）

由于我的卫星图像要现在NOAA网站上下载HDF5原始格式的文件，然后再写代码画图（由于波段的问题还要调节对比度，不过我比较偷懒我将hdf5中的radiance数值乘了10^9），再将灰度图像转化为3通道图像（记住tensorflow的object_detection库一般只能做3通道图像的训练工作，当然您也可以自己写model完成灰度图像的训练），这样的过程不是每个人都有必要做的，所以我建议如果只是为了尝试训练的话，可以自己写一个爬虫在网上爬下一些图像数据，或者使用相关视频做间隔截图做我们的训练数据就好了。
附上一个截图代码给大家：
```
import cv2
cap = cv2.VIdeoCapture(“video.mp4”)
c = 1
timef = 100 	#每100帧保存一张图像
while True;
    rval,frame = cap.read()
    if(c % timef == 0):
		print(‘tot=’,tot)
		cv2imwrite(‘out/‘+str(tot).zfill(6)+’.jpg’,frame)		
        tot = tot +1
	c+=1
	cv2.waitKey(1)
cap.release()
```
这样我们的images文件夹的内容就大概准备好了，接下来解释图像标注的工作也就是要搞定annotations文件夹的内容，这里我的方案是使用labelImg做标注，大家可以直接在github上搜索[labelImg](https://github.com/tzutalin/labelImg)，这是一个图形化的图像标记工具，并且可以使用bounding box在图片中标注目标。下面说明一种最简单的安装方式：
确认是否已经安装了需要的依赖库，并直接安装，我的环境是ubuntu16.04 python3+Qt5，先将github repo clone到本地，然后执行：
```
sudo apt-get install pyqt-dev-tools
sudo pip3 install lxml
python3 labelImg.py
```
然后就是较为漫长的使用GUI界面进行标注，注意在标注之前选定好annotations存储的目录。我们可以看到生成的xml文件大概张这个样子：
```
<annotation>
	<folder>images</folder>
	<filename>0032.png</filename>
	<path>/home/gabrielsun/Desktop/images/0032.png</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>651</width>
		<height>520</height>
		<depth>1</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>waves</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>204</xmin>
			<ymin>170</ymin>
			<xmax>317</xmax>
			<ymax>291</ymax>
		</bndbox>
	</object>
	<object>
		<name>waves</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>355</xmin>
			<ymin>274</ymin>
			<xmax>423</xmax>
			<ymax>466</ymax>
		</bndbox>
	</object>
	<object>
		<name>waves</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>489</xmin>
			<ymin>168</ymin>
			<xmax>587</xmax>
			<ymax>298</ymax>
		</bndbox>
	</object>
	<object>
		<name>waves</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>290</xmin>
			<ymin>68</ymin>
			<xmax>427</xmax>
			<ymax>161</ymax>
		</bndbox>
	</object>
</annotation>

```
但是我们这样的pascal VOC格式的数据集是不能直接被object detection API所使用的，要先将数据转化成TFrecord格式。TFrecord文件是一种二进制文件，能更好的利用内存，更方便的复制和移动，并且不需要单独的标记文件，附上转换代码：
```
import os
import io
import xml.etree.ElementTree as ET
import tensorflow as tf

from object_detection.utils import dataset_util
from PIL import Image


flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('images_dir', '', 'Path to directory of images')
flags.DEFINE_string('labels_dir', '', 'Path to directory of labels')
FLAGS = flags.FLAGS


def create_tf_example(example):

    image_path = os.getcwd() + '/' + FLAGS.images_dir + example
    labels_path = os.getcwd() + '/' + FLAGS.labels_dir + \
        os.path.splitext(example)[0] + '.xml'

    # Read the image
    img = Image.open(image_path)
    width, height = img.size
    img_bytes = io.BytesIO()
    img.save(img_bytes, format=img.format)

    height = height
    width = width
    encoded_image_data = img_bytes.getvalue()
    image_format = img.format.encode('utf-8')

    # Read the label XML
    tree = ET.parse(labels_path)
    root = tree.getroot()
    xmins = xmaxs = ymins = ymaxs = list()

    for coordinate in root.find('object').iter('bndbox'):
        xmins = [int(coordinate.find('xmin').text) / width]
        xmaxs = [int(coordinate.find('xmax').text) / width]
        ymins = [int(coordinate.find('ymin').text) / height]
        ymaxs = [int(coordinate.find('ymax').text) / height]

    classes_text = ['waves'.encode('utf-8')]
    classes = [1]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(encoded_image_data),
        'image/source_id': dataset_util.bytes_feature(encoded_image_data),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),,
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    for filename in os.listdir(FLAGS.images_dir):
        tf_example = create_tf_example(filename)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()
```
运行:

```
python3 convert_labels_to_tfrecords.py \ 
--output_path=train.record \ 
--images_dir=path/to/your/training/images/ \
--labels_dir=path/to/your/training/label/xml/
```
这样training set的数据就已经转化为tfrecord格式了，再如法炮制将validation set的images和annotations转化为tfrecord并命名为val.record。这样我们的数据集就算是准备好了。
# 模型的训练和预测
在research目录下新建2018420目录作为我们的工作目录，如此命名主要是为了日后的版本管理方便。将之前生成的两个tfrecord文件放入到这个目录。另外将object_detection/data/pet_label_map.pbtxt,object_detection/samples/configs/faster_rfcnn_resnet152_pets.config这两个文件也copy到这个目录中。
我们不能直接使用config文件中的设置，要根据自己的情况进行调节。调节的超参数包括steps，learning rate等等。另外还要调节IO的文件地址。在我实践的过程中，我发现fine_tune_checkpoint的设置最好是空的，或者将from_detection_checkpoint参数设置为false。
接下来我们就可以开始训练了。先将目录切换到research文件夹下，然后执行:
```
python3 object_detection/train.py \ 
--logtostderr \ 
-- pipeline_config_path=‘./20180420/faster_rcnn_resnet152_pets.config’ \
--train_dir=‘./20180420/
```
这样正常情况下我们的训练过程就可以开始了。不过我在实际操作的过程中发现python3.5.2下安装tensorflow可能会出现bytes object与string转换的错误，这实际上是python2和python3在对字符运算上的不一致导致的。python2的string运算是将string先转化为bytes object在做运算，但是python3实际上是对string直接运算的，这里就需要我们在相关的代码中加入try、except了。不过我在加了相关的try、expect仍然会报错，而且错误栈实在是太长了，所以我偷了个懒，换了一种方法，直接使用python2下安装的tensorflow，使用anaconda可以直接转换tensorflow使用python>解释器的版本～
	
让我们先看看运行结果：
![运行结果](/img/object_detection/2.png)
在运行训练程序结束后可以使用tensorboard将训练的过程可视化出来：
![过程可视化](/img/object_detection/3.png)
后面我们也可以选择对训练好的网络进行评估：
```
python3 object_detection/eval.py \ 
--logtostderr \
--pipeline_config_path=‘./20180420/faster_rcnn_resnet152_pets.config’ \ 
--checkpojint_dir=‘./20180420‘
--eval_dir=‘./20180420’ 
```
在结尾之前我再稍稍讲一下模型的存储和加载，tensorflow提供了两种方式来储存和加载模型：
1.生成检查点文件（checkpoint file），扩展名一般是.ckpt，通过在tf.train_saver对象上调用Saver.save()生成。它包含权重和其他在程序中定义的变量，不包含图结构。如果需要在另一个程序中使用，需要重新创建图形结构，并告诉tensorflow如何处理权重。
2.生成图协议文件（graph proto file）,扩展名一般是.pb，这是一个二进制文件。用tf.train.write_graph()保存，只包含图结构，不包含权重，然后使用tf.import_graph_def()来加载图形。
也就是说模型存储和图存储两者是互为补充的，如果我们要建立一个检测算法的话，就不能不将模型取出并读取到我的检测代码中。这样我们可以介绍一下，生成模型.pb文件的方法：
```
python3 object_detection/export_inference_graph.py \ 
--input_type image_tensor
--pipeline_config_path=./2018420/faster_rcnn_resnet152_pets.config
--trained_checkpoint_prefix=./2018420/model.ckpt-1812 
--output_directory=./2018420/output_inference_graph 
```
附上我的预测代码：
```
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
# from matplotlib import pyplot as plt ### CWH
from PIL import Image

# if tf.__version__ != '1.4.0':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')

# ENV SETUP  ### CWH: remove matplot display and manually add paths to references
'''
# This is needed to display the images.
%matplotlib inline
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
'''

# Object detection imports

from object_detection.utils import label_map_util  # CWH: Add object_detection path

# from object_detection.utils import visualization_utils as vis_util ### CWH: used for visualization

# Model Preparation

# What model to download.
# MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_FILE = MODEL_NAME + '.tar.gz'
# DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = '/home/gabrielsun/tensorflow_python2.7/models/research/2018422/output_inference_graph/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('/home/gabrielsun/tensorflow_python2.7/models/research/2018422',
                              'pet_label_map.pbtxt')  # CWH: Add object_detection path

NUM_CLASSES = 1


# Download Model
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())


# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# Detection
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = '/home/gabrielsun/Documents/Graduation_Design/code_and_data/dataset_train/images/'  # cwh
TEST_IMAGE_PATHS = [os.path.join(
    PATH_TO_TEST_IMAGES_DIR, '00{}.png'.format(i)) for i in range(45, 46)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      # CWH: below is used for visualizing with Matplot
      '''
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(image_np)  
      '''

      # CWH: Print the object details to the console instead of visualizing them with the code above

      classes = np.squeeze(classes).astype(np.int32)
      scores = np.squeeze(scores)
      boxes = np.squeeze(boxes)
      # print(scores)
      threshold = 0.20  # CWH: set a minimum score threshold of 50%
      obj_above_thresh = sum(n > threshold for n in scores)

      print("detected %s objects in %s above a %s score" %
            (obj_above_thresh, image_path, threshold))
      for c in range(0, len(classes)):
        if scores[c] > threshold:
          class_name = category_index[classes[c]]['name']
          print(" object %s is a %s - score: %s, location: %s" %
                (c, class_name, scores[c], boxes[c]))

```
总结一下这次时间的过程，这次实践实际上难点是在程序执行环境的配置上，对于训练算法的应用不是很明显。所以我会在后面的blog中弥补这些内容。另外我也由于这次的机会找到了一个难得的源代码学习的途径—-就是google的object detection API。首先，它是在适应工程的基础上高度标准化的，对于只接触过ML皮毛的人来说这也可以靠修改网络的config文件进行得心应手的训练。另外由于google在云端计算上的突出表现我们的计算过程也是可以放在cloud上进行的，而且这样的过程也是可视化的，并且我们的关注点将更少的放在系统层甚至是硬件问题上，更多的是去讨论算法上的核心问题。
好了，这次的实践内容告一段落，有问题可以在评论区找到我哦～

<!--more-->
转载请注明出处: https://suncol.netlify.app/