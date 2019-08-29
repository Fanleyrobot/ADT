#!/usr/bin/env bash

######################################################################
# Torch install
######################################################################


TOPDIR="$(dirname "$PWD")"

# Prefix:
PREFIX=$TOPDIR/torch
echo "Downloading luarocks: ..."

if [[ `uname` != 'Linux' ]]; then
  echo 'Platform unsupported, only available for Linux'
  exit
fi
if [[ `which apt-get` == '' ]]; then
    echo 'apt-get not found, platform not supported'
    exit
fi

# Download base packages:
$PREFIX/bin/luarocks download --rockspec cwrap
$PREFIX/bin/luarocks download --rockspec paths
$PREFIX/bin/luarocks download --rockspec torch
$PREFIX/bin/luarocks download --rockspec luaffi
$PREFIX/bin/luarocks download --rockspec nn
$PREFIX/bin/luarocks download --rockspec cutorch
$PREFIX/bin/luarocks download --rockspec cunn
$PREFIX/bin/luarocks download --rockspec luafilesystem
$PREFIX/bin/luarocks download --rockspec penlight
$PREFIX/bin/luarocks download --rockspec sys
$PREFIX/bin/luarocks download --rockspec xlua
$PREFIX/bin/luarocks download --rockspec image
$PREFIX/bin/luarocks download --rockspec camera
$PREFIX/bin/luarocks download --rockspec env
$PREFIX/bin/luarocks download --rockspec qtlua
$PREFIX/bin/luarocks download --rockspec qttorch
$PREFIX/bin/luarocks download --rockspec gnuplot
$PREFIX/bin/luarocks download --rockspec nngraph

$PREFIX/bin/luarocks download --rockspec sundown
$PREFIX/bin/luarocks download --rockspec dok
$PREFIX/bin/luarocks download --rockspec camera
$PREFIX/bin/luarocks download --rockspec graph

echo "Packaging luarocks: ..."
# Pack base packages:
$PREFIX/bin/luarocks pack cwrap-*.rockspec
$PREFIX/bin/luarocks pack paths-*.rockspec
$PREFIX/bin/luarocks pack torch-*.rockspec
$PREFIX/bin/luarocks pack luaffi-*.rockspec
$PREFIX/bin/luarocks pack nn-*.rockspec
$PREFIX/bin/luarocks pack cutorch-*.rockspec
$PREFIX/bin/luarocks pack cunn-*.rockspec
$PREFIX/bin/luarocks pack luafilesystem-*.rockspec
$PREFIX/bin/luarocks pack penlight-*.rockspec
$PREFIX/bin/luarocks pack sys-*.rockspec
$PREFIX/bin/luarocks pack xlua-*.rockspec
$PREFIX/bin/luarocks pack image-*.rockspec
$PREFIX/bin/luarocks pack camera-*.rockspec
$PREFIX/bin/luarocks pack env-*.rockspec
$PREFIX/bin/luarocks pack qtlua-*.rockspec
$PREFIX/bin/luarocks pack qttorch-*.rockspec
$PREFIX/bin/luarocks pack gnuplot-*.rockspec
$PREFIX/bin/luarocks pack nngraph-*.rockspec

$PREFIX/bin/luarocks pack sundown-*.rockspec
$PREFIX/bin/luarocks pack dok-*.rockspec
$PREFIX/bin/luarocks pack camera-*.rockspec
$PREFIX/bin/luarocks pack graph-*.rockspec

echo "Compressing luarocks into LuaRocks.zip ..."
zip LuaRocks.zip *.src.rock
echo "Deleting all src rocks and rock specs ..."
rm *.src.rock
rm *.rockspec

echo ""
echo "=> Luarocks have been downloaded, packed and compressed successfully."
echo "File: LuaRocks.zip"
echo ""
