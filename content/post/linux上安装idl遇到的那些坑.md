---
author: "Cong Sun"
title: "Linux上安装idl遇到的那些坑"
date: 2020-05-21T19:47:26+08:00
lastmod: 2020-05-21T19:47:26+08:00
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

# 写在前面
最近作者所在的实验室服务器遇到了一些配置问题。一开始，是我的老师和我讲服务器的idl无法调出idlde的图形化界面，在后来的测试中我发现确实无法正常调出图形界面，而问题的原因则十分有趣。是我之前在安装idl的时候为了图方便(其实还是现在idl的破解资源比较难找)安装了idl7.1的32位版，导致实验室的64位机器运行异常。在尝试安装linux的32位库补救失败后，我打算安装正常的64位软件，no matter it cost。

<!--more-->

# software & crack file
在万能的google帮助下我获得了idl83的安装包和配置文件，这里分享下：
链接: https://pan.baidu.com/s/1NFaD5W8z5jL1tZVYpf3T1Q 密码: e6xq

# install
将上面的安装包下载到本地，并解压：
```
tar -xvf idl83envi51linux.x86_64.tar.gz
```

等待下idl就在当前的文件夹中被解压出来了，这时候找到install.sh脚本文件，执行该文件：

```
sudo sh install.sh //这里最好使用sudo用户安装或者相应文件夹下的权限
```
安装脚本执行的时候会提示，安装过程中会遇到许可说明，长按回车到结束（或按q直接跳出），遇到询问可以输入y回车，其中有要输入安装路径（Please enter the directory to contain IDL 8.2），填一个想要安装的位置(如果位置文件夹不存在，需要提前建立)，这个安装位置我选择的是默认位置，最好将这个安装位置记一下，以后出现问题的时候便于查找。

最后一个问题是是否运行license引导程序（Do you want to run the License Wizard? (y/n): y），输入y回车，程序会弹出对话框，选择Install a license you rescived。然后先不要在这里做操作，我们开始执行破解操作的部分。

首先给出linux下idl83的license.dat文件：

```
############ license file comments, do not delete ###############  
# License Number(s):231821-3  
SERVER cga2.cga.harvard.edu 089e01ba034b 1700  
DAEMON idl_lmgrd  
INCREMENT envi idl_lmgrd 5.100 1-jan-0000 5 585C4CBDAFB11CB5 \  
    VENDOR_STRING="231821-3Harvard University (MAIN)" ck=121 \  
    SIGN="0B17 EFA5 CA84 0013 A7A7 7A29 8D4B 0EF6 A4FE 8377 EB66 \  
    447F F317 C8E1 F65D 1992 9E0E 4381 C14D 5D0F 9593 4519 135E \  
    1BF4 D28C 111F 19FD F592 DC04 D365"  
INCREMENT envi_cartosat idl_lmgrd 5.100 1-jan-0000 5 
1BDE84587C8E9E79 \  
    VENDOR_STRING="231821-3Harvard University (MAIN)" ck=98 \  
    SIGN="053A F301 5887 ADF7 4C68 67FE E006 A4FB 4BDE 34A1 83AA \  
    241F DE2C 3229 C9BF 1685 58A4 12BB 98F2 DC32 0518 E29D 7C82 \  
    81C7 B477 7725 0D1E E041 89E7 B4DB"  
INCREMENT idl idl_lmgrd 8.300 1-jan-0000 30 025958CE707A165B \  
    VENDOR_STRING="231821-3Harvard University (MAIN)" ck=108 \  
    SIGN="1605 4794 E80A BFC5 3E7E D228 F4D4 9BB2 01D3 1AE7 768B \  
    7568 E3D7 ED79 07AA 0F04 8E3D 649F 788D 050C 4CA0 17B0 8678 \  
    4CDF F517 A300 7C72 0A5D 23E5 CB4F"  
INCREMENT idl_bridge_assist idl_lmgrd 8.300 1-jan-0000 5 \  
    58DA4BE13A8CFD8B VENDOR_STRING="231821-3Harvard University \  
    (MAIN)" ck=167 SIGN="1823 FEE3 AC65 6C32 0C04 FD1A 6503 1901 \  
    DE8D EFAF A3EA EE98 1E0D 3D13 1DAF 0431 5467 626F AA26 284A \  
    925B 7BF9 763B 6641 1B6A 2A71 A934 4176 B8F5 EF94"  
INCREMENT idl_video_write idl_lmgrd 8.300 1-jan-0000 5 \  
    18322A8CF2B5C47F VENDOR_STRING="231821-3Harvard University \  
    (MAIN)" ck=84 SIGN="1810 BF0E 3A36 AFD6 6B8A 2DAE CCA4 DC50 \  
    B195 2166 82E0 1EA4 FB96 394F 79D5 0ECA C0AB C13D B4EB 7F31 \  
    AE14 C099 E62F 7790 CF97 A2B5 568C 6EF4 5440 5B93"  
INCREMENT idl_wavelet idl_lmgrd 8.300 1-jan-0000 5 
0BFC71E479FA6A91 \  
    VENDOR_STRING="231821-3Harvard University (MAIN)" ck=157 \  
    SIGN="0FA1 3E4D 9FC1 8267 FB86 6953 7E8D CE58 CE37 DB44 5941 \  
    48A6 2255 CE60 D325 0751 D013 55BB 3C55 17AB 3C63 9C09 3059 \  
    CB98 D1B2 B04F CB01 A2E9 7CA4 4755"  
FEATURESET idl_lmgrd C107256B542AC2F8  
  
##################### end of license file #######################
```

将上面正文第一行SERVER后面的cga2.cga.harvard.edu改为自己机器的主机名，查询主机名可以使用:
```
hostname -a
```
将主机名更改后，需要修改系统的MAC地址与license.dat文件中的地址相对应。licens.dat文件中MAC的值就在hostname的后面，也就是089e01ba034b这部分。
在终端下输入：

```
sudo ifconfig eth0 hw ether 08:9e:01:ba:03:4b
```
可以临时修改MAC地址，在终端下输入：
```
ifconfig
```
发现输出对应的网卡下MAC地址值为上述的修改值则操作成功。

这时回到license引导程序，将license.dat文件放置在安装文件夹下的license文件夹中（我的例子中该文件夹的位置是/usr/local/exelis/license），并检查文件夹和license.dat文件的是否有执行权限（r+x权限）。然后将license引导程序的license位置在其中载入，点击next。
然后选择Install license manager; Start license manager。
到这里安装过程结束。

# 检查是否安装成功
终端输入idl
```
idl
```
得到输出结果：
```
(base) [root@localhost license]# idl
IDL Version 8.3 (linux x86_64 m64). (c) 2013, Exelis Visual Information Solutions, Inc.
Installation number: 231821-3.
Licensed for use by: Harvard University (MAIN)

IDL>
```
则安装成功。

# 问题及解决

## 缺少lmgrd 报错：error: “lmgrd: not found”
在安装的结束后运行idl发现license manager有错误，提示没有启动lmgrd，但是在/usr/local/exelis/idl/bin文件夹中查找的时候，发现没有lmgrd这个可执行文件。
参考：https://www.harrisgeospatial.com/Support/Self-Help-Tools/Help-Articles/Help-Articles-Detail/ArtMID/10220/ArticleID/16128/Running-IDL-83-and-above-lmgrd-fails-with-error-lmgrd-not-found
我安装了lsb-core之后，再cd到上述文件夹下就发现了lmgrd文件，执行该可执行文件（可能需要sudo）。
```
./lmgrd
```
然后再执行：
```
idl
```
问题解决。

## 重启后破解失效
解决该问题的方法是每次重启或断网后执行：
```
ifconfig eth0 hw ether 08:9e:01:ba:03:4b
cd /usr/local/exelis/idl/bin
./lmgrd
```
或者直接将上述过程写入开机脚本中：
```
sudo gedit /etc/rc.local
//将以下三行代码加到exit 0的前面，然后保存
ifconfig eth0 down
ifconfig eth0 hw ether 08:9e:01:ba:03:4b
ifconfig eth0 up
```
亲测上述的两种方法各有优劣，第一种方法比较麻烦每次开机的时候都要进行一次，而第二种方法会造成网络不稳定，有可能会开机后断网。请视情况选择。

最后不得不说的就是服务器的这些问题就是不要怕试错，经验的积累都是在不断的尝试过程中展开的，希望自己可以保持学习的心~

<!--more-->
转载请注明出处: https://suncol.netlify.app/