#!/usr/bin/env bash

######################################################################
# Torch install
######################################################################


#TOPDIR=$PWD
TOPDIR=$HOME

# Prefix:
#PREFIX=$PWD/torch
PREFIX=$HOME/torch
echo "Installing Torch into: $PREFIX"

if [[ `uname` != 'Linux' ]]; then
  echo 'Platform unsupported, only available for Linux'
  exit
fi
if [[ `which apt-get` == '' ]]; then
    echo 'apt-get not found, platform not supported'
    exit
fi

# Install dependencies for Torch:
sudo apt-get update
sudo apt-get install -qqy build-essential
sudo apt-get install -qqy gcc g++
sudo apt-get install -qqy cmake
sudo apt-get install -qqy curl
sudo apt-get install -qqy libreadline-dev
sudo apt-get install -qqy git-core
sudo apt-get install -qqy libjpeg-dev
sudo apt-get install -qqy libpng-dev
sudo apt-get install -qqy ncurses-dev
sudo apt-get install -qqy imagemagick
sudo apt-get install -qqy unzip
sudo apt-get install -qqy gnuplot
sudo apt-get install -qqy qt4-qmake
sudo apt-get install -qqy libqt4-dev
sudo apt-get update


echo "==> Torch7's dependencies have been installed"



# Build and install Torch7
cd /tmp
rm -rf luajit-rocks
git clone https://github.com/torch/luajit-rocks.git
cd luajit-rocks
mkdir -p build
cd build
git checkout master; git pull
rm -f CMakeCache.txt
cmake .. -DCMAKE_INSTALL_PREFIX=$PREFIX -DCMAKE_BUILD_TYPE=Release
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
make
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
make install
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi


path_to_nvcc=$(which nvcc)
if [ -x "$path_to_nvcc" ]
then
    cutorch=ok
    cunn=ok
fi

# Install base packages:
sudo $PREFIX/bin/luarocks install cwrap
sudo $PREFIX/bin/luarocks install paths
sudo $PREFIX/bin/luarocks install torch
sudo $PREFIX/bin/luarocks install nn

[ -n "$cutorch" ] && \
(sudo $PREFIX/bin/luarocks install cutorch)
[ -n "$cunn" ] && \
(sudo $PREFIX/bin/luarocks install cunn)

sudo $PREFIX/bin/luarocks install luafilesystem
sudo $PREFIX/bin/luarocks install penlight
sudo $PREFIX/bin/luarocks install sys
sudo $PREFIX/bin/luarocks install xlua
sudo $PREFIX/bin/luarocks install image
sudo $PREFIX/bin/luarocks install camera
sudo $PREFIX/bin/luarocks install env
sudo $PREFIX/bin/luarocks install qtlua
sudo $PREFIX/bin/luarocks install qttorch
sudo $PREFIX/bin/luarocks install gnuplot

echo ""
echo "=> Torch7 has been installed successfully"
echo ""


echo "Installing nngraph ... "
sudo $PREFIX/bin/luarocks install nngraph
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "nngraph installation completed"

echo "export TORCH_PATH=$PREFIX" >> "$HOME"/.bashrc
export TORCH_PATH=$PREFIX

echo "Installing lunatic-python ... "
cd $TOPDIR
git clone https://FangyiZhang@bitbucket.org/FangyiZhang/lunatic-python.git
cd lunatic-python
./install.sh
RET=$?; if [ $RET -ne 0 ]; then echo "Error. Exiting."; exit $RET; fi
echo "Lunatic-python installation completed"

echo
echo "You can run experiments by executing: "
echo
echo "   ./deeprobot_cpu task_name"
echo
echo "            or   "
echo
echo "   ./deeprobot_gpu task_name"
echo
