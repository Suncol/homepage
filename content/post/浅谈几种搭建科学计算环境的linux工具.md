---
author: "Cong Sun"
title: "浅谈几种搭建科学计算环境的linux工具"
date: 2020-05-21T21:24:13+08:00
lastmod: 2020-05-21T21:24:13+08:00
draft: false
description: ""

tags: ["科学计算环境"]
categories: ["技术"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: false
auto_collapse_toc: true
math: false
---

现在做科学计算相关的工具有很多。除了大多数时候用在超算上的module环境管理之外，也有很多有趣的软件。而且并不是所有人所有时候都可以使用超算，超算也并不是科学计算的唯一硬件解决方案。我写这个文章之前，尝试管理我们组的服务器环境有一年的时间了，其中run过4，5个不同的模式，也算是在搭建环境上有点心得，正好今天总结分享下。

<!--more-->

一开始我在搭建环境的时候，就是简单的有pre-build包就直接安装，没有就源代码编译，走的耿直路线，但是很快就发现，随着服务器库环境的逐渐复杂，这样的作法会使得整个环境变量混论不堪，而且每次还要去找对应的库编译安装的位置，也是非常的耗时耗力。
后来我找到了4中比较好用且相对成熟的服务器环境管理工具(这里不提module，毕竟我主要使用的是组里的一个intel6154工作站，不是集群)。

## conda
首先是[conda](https://docs.conda.io/en/latest/),这里放在第一个介绍是因为我的包管理器的启蒙就是它。使用非常方便简洁，在日常的python使用中基本上涵盖了所有常用库，而且还有[intelpython](https://software.intel.com/en-us/distribution-for-python)的神秘加成。推荐使用的时候对于不同的项目都新建一个env，不仅仅可以使开发的环境相对独立，而且在未来的给他人介绍自己写的代码的时候，也可以方便的分享依赖库(conda list -e > requirements.yml )。对于主要使用python开发的项目，十分推荐。

## spack
而后是[spack](https://spack.io/),这是一个可支持不同平台的包管理系统。开发的主要目的就是为了更方便的安装科学计算所需要的库，并且支持不同编译器/编译器版本之间的切换，可以说是十分方便。spack的使用逻辑和conda很像，这也是我把spack放在第二个讲的原因。spack安装非常方便，只要求host上有python环境，而后直接按照spack doc上的说明就可以了。spack install可以安装库，而且使用一定的语法指定库的特性，如spack install hdf5%gcc7 就表示使用spack安装gcc编译的hdf5库，在不添加pre-build mirror的条件下，整个安装过程可以理解为先下载对应的库和依赖的源码包然后按照指定的编译方式进行编译安装，这里只是简单的举了一个例子，实际使用的时候会有更多灵活的性质。安装好的spack库会存放在spack_root下的/opt目录中，与linux日常使用的/opt目录类似，install的库使用hash散列，方便查找以及彼此区分。在使用的时候我们可以直接使用库名——Tab查看选择，或者直接使用库的hash值进行类似module load的操作(这里是spack load)。所以虽然spack也有类似于conda env create的功能，但是一般情况下spack load就可以满足大部分的需要了，无需特殊使用。但是我也要泼泼冷水，spack虽然功能好用，但是在实际使用的时候，库安装的时间还是很感人的，毕竟是和源码安装类似的过程，build时间占了很长的部分，所以强烈建议使用pre-build mirror。但是有的时候mirror中没有对应的库，那就只能慢慢install了。另外，spakc的安装对于网络的稳定要求比较高，而且在结界之内使用体验较差，最好搭配相应的工具使用。

## docker/singularity
下一个就是[docker](https://www.docker.com/)/[singularity](https://sylabs.io/docs/)。前者可能大佬们都听说过了，docker是进程级的独立环境，早期是由linux容器实现的，所以在linux机器中可以使用GPU。但是win和mac的机器中，docker 容器和驱动之前隔了一层虚拟机，因此无法直接使用GPU等加速设备。不要问为啥mac也是要靠一层虚拟机，我只能说不是所有满足Unix标准的都是好os(玩笑)。docker是一般情况下，最好的环境配置方案，只要可以在docker hub上找到对应的docker image，直接pull下来，然后启动容器就可以直接测试环境了。由于是对于含GPU科学计算应用，只需要host的驱动较高，可以向下兼容一定的cuda版本，那么使用docker管理不同的cuda环境是最经济的方式。之前我不了解docker的应用，在host上装了多个版本的cuda，直接导致host驱动gg，这都是血的教训！但是docker也有不太ok的地方，首先就是上手比较麻烦，在一开始的操作中容易image和container傻傻分不清，引起一些错误。另外就是docker bind folder的时候会有权限的问题，这个问题对于一般的使用可以接受，但是使用久了还是蛮麻烦的，所以我这里推荐使用singularity，它可以看作是一个为超算环境打造的docker，对mpi/openmp等并行库的支持比较好，而且支持在singularity环境中使用环境$user，减少了很多文件权限的问题，使用之后感觉在科学计算这个场景上是更优的选择。而且singularity的很多操作是面向文件的，我们很多时候只需要在它的[images market](https://cloud.sylabs.io/library)上直接下载sif文件(当然也可以使用singularity pull，但是网速你懂的)然后直接singularity shell *.sif就可以进入环境了，配合--writable option可以修改容器环境，也可以在完成对容器环境的修改之后重新build一个新sif file用于环境分发。不可谓不方便。而且singularity和spack类似，不需要管理员权限，就可以自行编译安装，以及运行环境，可以让超算用户也获得较为灵活的环境控制。(docker是不能的，因为它需要root start一个守护进程！)
在未来的科学计算应用中，我的倾向是使用后面的容器技术做环境的分发。毕竟对于很多科学模式/程序，库依赖关系错综复杂，很多研究人员的青春和头发都献给了无聊的环境搭建工作，有了容器这个利器，只需要在发布相应版本的计算代码的时候同时发布对应的images即可，对于非异构的项目甚至可以做到多平台分发，大大的增加的科研flow的效率。自己立下个flag，'origin -flag ->' !

<!--more-->
转载请注明出处: https://suncol.netlify.app/