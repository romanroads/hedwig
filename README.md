<p align="center">
  <img src="artifacts/ROMAN_ROADS_LOGO_COLOR.png" width="500" title="ROMAN ROADS, INC.">
</p>


# Project Hedwig

## Detection and tracking road agents using drone recorded video data

### Setup

- Tested OS
```
Linux 5.4.0-132-generic #148~18.04.1-Ubuntu SMP Mon Oct 24 20:41:14 UTC 2022 x86_64 x86_64 x86_64 GNU/Linux
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=18.04
DISTRIB_CODENAME=bionic
DISTRIB_DESCRIPTION="Ubuntu 18.04.3 LTS"
NAME="Ubuntu"
VERSION="18.04.3 LTS (Bionic Beaver)"
ID=ubuntu
ID_LIKE=debian
PRETTY_NAME="Ubuntu 18.04.3 LTS"
VERSION_ID="18.04"
HOME_URL="https://www.ubuntu.com/"
SUPPORT_URL="https://help.ubuntu.com/"
BUG_REPORT_URL="https://bugs.launchpad.net/ubuntu/"
PRIVACY_POLICY_URL="https://www.ubuntu.com/legal/terms-and-policies/privacy-policy"
VERSION_CODENAME=bionic
UBUNTU_CODENAME=bionic

```

- Install miniconda
  - conda >= 4.8.2
- Create virtual environment
  - conda env create --file python/environment_ubuntu.yml
  - conda activate hedwig
  - ./shell_scripts/setup.sh
  

### Pre-trained model

Download pre-trained model here: https://behavior-data-open-source.s3.cn-north-1.amazonaws.com.cn/model_rr_net.pth 
place the model file under (create a folder named "data" under your repo folder, this data folder is git-ignored)
```
./data/model_rr_net.pth
```

### Test video data

Download pre-trained model here: https://behavior-data-open-source.s3.cn-north-1.amazonaws.com.cn/test.mp4 
place the model file under
```
./data/test.mp4
```

### How to run the pipeline

Run the following script under repo folder

```
./shell_scripts/launch.sh
```

- It will ask you to select a ROI region first, point & click a few points that covers the road area, and type "d" when done
- Do similar point & click for registration zone, where new cars can be registered, type "n" key if you have multiple registration zones
- Do similar point & click for un-subscription zone where cars can be removed from system
- Occlusion and stablization zones are for those areas there are trees and billboards blocking the vehicles
- Type "ESC" key to quit the process

It will dump out the csv, txt file ("_mvd.csv", "_ROI.txt") as wel as the annotated video file ("_processed.mp4") under ./data/

### Offline analysis code

ALl offline analysis such as figuring out which vehicle is in front of which vehicle, so on and so forth, are located at

```
./python/offline_analysis/
```