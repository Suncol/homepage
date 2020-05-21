---
author: "Cong Sun"
title: "服务器安装netcdf库说明"
date: 2020-05-21T21:28:01+08:00
lastmod: 2020-05-21T21:28:01+08:00
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

本文参照[博客](https://www.jianshu.com/p/90ecc0580bd1)的讲述，介绍在linux服务器上安装netcdf的一般方法。主要的目的是实现数据压缩的功能，其他详细的介绍可以参考[官方指南](https://www.unidata.ucar.edu/software/netcdf/docs/getting_and_building_netcdf.html)。

<!--more-->

# 获取编译所需源码
首先安装所需要的包实现Data Compression所必须的包只有四个：
- [HDF5 1.8.9 or later (for netCDF-4 support)](https://link.jianshu.com/?t=ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/)
- [zlib 1.2.5 or later (for netCDF-4 compression)](https://link.jianshu.com/?t=ftp://ftp.unidata.ucar.edu/pub/netcdf/netcdf-4/)
- [netcdfc (latest version)](https://link.jianshu.com/?t=https://github.com/Unidata/netcdf-c/releases/v4.4.1)
- [netcdf-fortran(latest version)](https://link.jianshu.com/?t=https://github.com/Unidata/netcdf-fortran/releases/v4.4.4)
 
另：非必装：curl 7.18.0 or later (for DAP remote access client support)

# 安装
由于一般服务器中可能会同时共存很多编译器，这里首先指定使用intel编译器编译：

# 指定编译器
对于linux：
```
export CC=icc
export CXX=icpc
export CFLAGS='-O3 -xHost -ip -no-prec-div -static-intel'
export CXXFLAGS='-O3 -xHost -ip -no-prec-div -static-intel'
export F77=ifort
export FC=ifort
export F90=ifort
export FFLAGS='-O3 -xHost -ip -no-prec-div -static-intel'
export CPP='icc -E'
export CXXCPP='icpc -E'
```

对于mac：
```
export CC=icc
export CXX=icpc
export CFLAGS='-O3 -xHost -ip -no-prec-div'
export CXXFLAGS='-O3 -xHost -ip -no-prec-div'
export F77=ifort
export FFLAGS='-O3 -xHost -ip -no-prec-div -mdynamic-no-pic'
```

# zlib安装
这里为了后续便于管理，所以选择将这些包都编译安装在/usr/local/文件夹下,因此也需要sudo权限
```
ZDIR=/usr/local/zlib    #安装文件夹
mkdir ${ZDIR}
./configure --prefix=${ZDIR}
make check
make install
```

# hdf5安装
```
H5DIR=/usr/local/hdf5
mkdir ${H5DIR}
./configure --with-zlib=${ZDIR} --prefix=${H5DIR} -enable-fortran -enable-cxx
make check
make install
```
这里需要注意的是，在安装hdf5时，必须要有–with-zlib=${ZDIR}，否则会报错相关库文件找不到。并且在个人PC机上安装时，可能会提示C++的注释方式不可用于IS90规范中，遇此错误，目前的解决方法是找到相关文件，将C++的注释方式//改为C的注释方式/ /，目前已知安装过程中有两个文件共两处需要修改。
更多内容可以参考[hdf5相关配置选项](https://link.jianshu.com/?t=https://support.hdfgroup.org/HDF5/release/chgcmkbuild.html)

# hdf4安装
```
在实际的安装过程中，由于程序需要同时使用hdf4和hdf5的库，因此这里给出hdf4的安装过程。
在安装hdf4之前，需要安装依赖库，jpeg,szip
```

## jpeg安装
在按照上面所提到的过程制定编译器之后，先到[jpeg的网站](http://libjpeg.sourceforge.net/)下载源代码包，解压后执行：
```
JPEGDIR=/usr/local/jpeg
mkdir ${JPEGDIR}
./configure --prefix=${JPEGDIR}
make check
make install
```

## szip安装
```
SDIR = /usr/local/szip
mkdir ${SDIR}
./configure --prefix=${SDIR}
make check
make install
```

## 编译安装hdf4
在这一步的时候有可能会遇到一些问题，需要注意./configure 的参数。在执行前建议先使用：
```
./configure --help
```

下载并解压hdf4安装包之后，请先确定服务器中是否安装了编译所需要的yacc和flex,若没有，请执行下列代码进行安装:
```
# ubuntu 
sudo apt-get install byacc
sudo apt-get install flex

# centos
sudo yum instal byacc
sudo yum install flex
```
满足了基本的编译条件之后，执行:
```
mkdir /usr/local/hdf4
./configure --with-szip=/usr/local/szip --with-jpeg=/usr/local/jpeg --with-zlib=/usr/local/zlib  --disable-netcdf --enable-fortran --prefix=/usr/local/hdf4
make check   #在对source code进行test的时候发现在hdfnctest这个测试中花费大量的时间，但是在中断make check直接make install居然成功了
make install
```

详细的参数请参考netcdf的[官方介绍](https://www.unidata.ucar.edu/software/netcdf/docs/getting_and_building_netcdf.html#build_hdf4)

# 安装netcdfc
```
NCDIR=/usr/local/netcdf4c
H5DIR=/usr/local/hdf5
H4DIR=/usr/local/hdf4
mkdir ${NCDIR}
CPPFLAGS=-I${H5DIR}/include LDFLAGS=-L${H5DIR}/lib 
CPPFLAGS=-I${H4DIR}/include LDFLAGS=-L${H4DIR}/lib
export LD_LIBRARY_PATH=${H5DIR}/lib:${LD_LIBRARY_PATH}
./configure --prefix=${NCDIR} --enable-netcdf-4 --enable-largefile --disable-dap  #这里在加入 --enable-hdf4参数之后始终会报错，错误为找不到hdf4 library，但是可以看到我事先已经将hdf4 lib加入到了环境变量中 这里存疑 但是不需要netcdf支持hdf4 因此可以忽略这点 
make check
make install
```

# 安装netcdff
```
NFDIR=/usr/local/netcdf4f
NCDIR=/usr/local/netcdf4c
mkdir ${NFDIR}
export CPPFLAGS="-I/usr/local/szip/include -I/usr/local/zlib/include -I/usr/local/hdf5/include -I/usr/local/netcdf4c/include"
export LDFLAGS="-L/usr/local/szip/lib -L/usr/local/zlib/lib -L/usr/local/hdf5/lib -L/usr/local/netcdf4c/lib"
export LD_LIBRARY_PATH=${NCDIR}/lib:${LD_LIBRARY_PATH}
CPPFLAGS=-I${NCDIR}/include LDFLAGS=-L${NCDIR}/lib ./configure --prefix=${NFDIR} --disable-fortran-type-check
make check
make install
```

# 安装结果
```
nc-config -all
```

# 注意事项
由于以上代码执行的时候是需要sudo权限的，但是使用sudo执行的时候会重置环境变量，所以导致了一些错误。建议安装的时候使用sudo -E
```
#-E选项在man page中的解释是
-E

The -E (preserve environment) option indicates to the security policy that the user wishes to preserve their existing environment variables. The security policy may return an error if the -E option is specified and the user does not have permission to preserve the environment.
```

简单来说，就是加上-E选项后，用户可以在sudo执行时保留当前用户已存在的环境变量，不会被sudo重置，另外，如果用户对于指定的环境变量没有权限，则会报错。

但是实际安装的时候发现，由于intel编译器环境变量的问题，使得make install的时候发生icpc command not found错误，因此我认为最佳的方法是事先将编译的目标文件夹权限设置好，然后直接执行上面的脚本，尽量避免使用sudo权限。另外在安装的过程中，我发现上述的几个库有些在编译链接的时候由于版本问题不能正常编译，这里给出我的编译包版本：
- szip-2.1.1.tar.gz
- zlib-1.2.11.tar.gz
- hdf5-1.10.4.tar.gz
- jpeg-9c-droppatch.tar.gz
- szip-2.1.1.tar.gz
- hdf-4.2.14.tar.gz
- netcdf-4.6.1.tar
- netcdf-fortran-4.4.4.tar.gz

<!--more-->
转载请注明出处: https://suncol.netlify.app/