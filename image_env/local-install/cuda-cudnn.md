## install cuda
```
sudo bash cuda_11.8.0_520.61.05_linux.run --toolkitpath=/mnt/sano1/tzw/local/cuda_118
```

result is

```
[sudo] password for tzw:
===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /mnt/sano1/tzw/local/cuda_118/

Please make sure that
 -   PATH includes /mnt/sano1/tzw/local/cuda_118/bin
 -   LD_LIBRARY_PATH includes /mnt/sano1/tzw/local/cuda_118/lib64, or, add /mnt/sano1/tzw/local/cuda_118/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /mnt/sano1/tzw/local/cuda_118/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 520.00 is required for CUDA 11.8 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log
```

## install cudnn

refer:

```
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
```

0. download

```
https://developer.nvidia.com/rdp/cudnn-archive
```

1. refer

```bash
https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html
```

3.

```
$ tar -xvf cudnn-linux-$arch-8.x.x.x_cudaX.Y-archive.tar.xz

```

4.

```bash

$ sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
$ sudo cp -P cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```