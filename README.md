# SBAS_InSAR_PyGMTSAR

#安裝GMTSAR

##Installation of GMT and GMTSAR with Homebrew

1.Download and install orbit files in /usr/local/orbits:
```
http://topex.ucsd.edu/gmtsar/tar/ORBITS.tar
sudo -i
cd /usr/local
mkdir orbits
cd orbits
tar -xvf ~/Downloads/ORBITS.tar # (need full path to ORBITS.tar)
```

sudo apt install csh subversion autoconf libtiff5-dev libhdf5-dev wget

sudo apt install liblapack-dev

sudo apt install gfortran

sudo apt install g++

sudo apt install libgmt-dev

sudo apt install gmt-dcw gmt-gshhg

sudo apt install gmt

sudo -i

cd /usr/local

git clone --branch 6.1 https://github.com/gmtsar/gmtsar GMTSAR

cd GMTSAR

autoconf

autoupdate

./configure --with-orbits-dir=/usr/local/orbits

make

make install

nano ./.bashrc

加入以下變量至最下方

export GMTSAR=/usr/local/GMTSAR

export GMTSAR_csh=/usr/local/GMTSAR/gmtsar/csh

export PATH=$GMTSAR/bin:"$PATH":$GMTSAR_csh 

按下ctl+x儲存

更新變量

source ~/.bashrc


# 創建虛擬機

pip install virtualenv

virtualenv -p python310 venv  (python 版本 3.10 up)

venv 是虛擬機名稱

windows venv:
    venv\Scripts\activate
    
mac venv:
    source venv\bin\activate

pip install -r requirements.txt

-- run python file --

python test1.py 
