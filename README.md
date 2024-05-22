#  Spike-based Neuromorphic Model for Sound Source Localization

<p align="center" float="center">
  <img src="https://github.com/i-spoon/Spike-based-Neuromorphic-Sound-Source-Localization/blob/main/Spike-based%20SSL%20model.png" width=80%/>
</p>

<p align="center" float="center">
  <img src="https://github.com/i-spoon/Spike-based-Neuromorphic-Sound-Source-Localization/blob/main/MAA.png" width=80%/>
</p>
## How to Run

First clone the repository.

```shell
git clone https://github.com/i-spoon/Spike-based-Neuromorphic-Sound-Source-Localization.git
cd src
pip install -r requirements.txt
```

### Train SSL model

Detailed usage of the script could be found in the source file.

```shell
python src/train.py
```

and the dataset folder `RF-PLC` should look like:

```
RF-PLC
├── snr50
│   ├── Training_1.5m_40channel_CQT_all_filter_4.mat
│   ├── Test_1.5m_40channel_CQT_all_filter_4.mat
├── snr20
│   ├── Training_1.5m_40channel_CQT_all_filter_4.mat
│   ├── Test_1.5m_40channel_CQT_all_filter_4.mat
├── snr0
│   ├── Training_1.5m_40channel_CQT_all_filter_4.mat
│   ├── Test_1.5m_40channel_CQT_all_filter_4.mat
...
```
