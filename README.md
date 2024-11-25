[stars-img]: https://img.shields.io/github/stars/yuntaoshou/LMAM?color=yellow
[stars-url]: https://github.com/yuntaoshou/LMAM/stargazers
[fork-img]: https://img.shields.io/github/forks/yuntaoshou/LMAM?color=lightblue&label=fork
[fork-url]: https://github.com/yuntaoshou/LMAM/network/members
[AKGR-url]: https://github.com/yuntaoshou/LMAM


# A Low-Rank Matching Attention Based Cross-Modal Feature Fusion Method for Conversational Emotion Recognition
![Supported Python versions](https://img.shields.io/badge/%20python-3.8-blue)
![Supported OS](https://img.shields.io/badge/%20Supported_OS-Windows-red)
[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
![image](https://github.com/yuntaoshou/CBERL/blob/main/fig/%E7%89%B9%E5%BE%81%E8%9E%8D%E5%90%88%2B%E6%95%B0%E6%8D%AE%E4%B8%8D%E5%B9%B3%E8%A1%A1.jpg)


This is an official implementation of 'A Low-Rank Matching Attention Based Cross-Modal Feature Fusion Method for Conversational Emotion Recognition' :fire:. Any problems, please contact shouyuntao@stu.xjtu.edu.cn. Any other interesting papers or codes are welcome. If you find this repository useful to your research or work, it is really appreciated to star this repository :heart:.

## Paper
[**A Low-Rank Matching Attention Based Cross-Modal Feature Fusion Method for Conversational Emotion Recognition**](https://arxiv.org/abs/2306.17799)<br>
Shou, Yuntao and Liu, Huan and Cao, Xiangyong and Meng, Deyu and Dong, Bo<br>
IEEE Transactions on Affective Computing, 2024

Please cite our paper if you find our work useful for your research:

```tex
@ARTICLE{10753050,
  author={Shou, Yuntao and Liu, Huan and Cao, Xiangyong and Meng, Deyu and Dong, Bo},
  journal={IEEE Transactions on Affective Computing}, 
  title={A Low-Rank Matching Attention Based Cross-Modal Feature Fusion Method for Conversational Emotion Recognition}, 
  year={2024},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TAFFC.2024.3498443}}
```

## Usage (Choose IEMOCAP-Six for Example)

### 🚀 Prerequisites
- Python 3.8
- CUDA 10.2
- pytorch ==1.8.0
- torchvision == 0.9.0
- torch_geometric == 2.0.1
- fairseq == 0.10.1
- transformers==4.5.1
- pandas == 1.2.5

(see requirements.txt for more details)


### Pretrained model

```shell
## for lexical feature extraction
https://huggingface.co/microsoft/deberta-large/tree/main  -> ../tools/transformers/deberta-large

## for acoustic feature extraction
https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt  -> ../tools/wav2vec

## for face extractor (OpenFace-win)
https://drive.google.com/file/d/1-O8epcTDYCrRUU_mtXgjrS3OWA4HTp0-/view?usp=share_link  -> ./OpenFace_2.2.0_win_x64

## for visual feature extraction
https://drive.google.com/file/d/1wT2h5sz22SaEL4YTBwTIB3WoL4HUvg5B/view?usp=share_link ->  ../tools/manet

## using ffmpeg for sub-video extraction
https://ffmpeg.org/download.html#build-linux ->  ../tools/ffmpeg-4.4.1-i686-static
```



### 🚀 Datasets

~~~~shell
# download IEMOCAP dataset and put it into ../emotion-data/IEMOCAP
https://sail.usc.edu/iemocap/iemocap_release.htm   ->   ../emotion-data/IEMOCAP

# whole video -> subvideo
python preprocess.py split_video_by_start_end_IEMOCAP

# subvideo -> detect face
python detect.py --model='face_detection_yunet_2021sep.onnx' --videofolder='dataset/IEMOCAP/subvideo' --save='dataset/IEMOCAP/subvideofaces' --dataset='IEMOCAP'

# extract visual features
cd feature_extraction/visual
python extract_manet_embedding.py --dataset='IEMOCAPFour' --gpu=0
python preprocess.py feature_compressed_iemocap dataset/IEMOCAP/features/manet dataset/IEMOCAP/features/manet_UTT

# extract acoustic features
python preprocess.py split_audio_from_video_16k 'dataset/IEMOCAP/subvideo' 'dataset/IEMOCAP/subaudio'
cd feature_extraction/audio
python extract_wav2vec_embedding.py --dataset='IEMOCAPFour' --feature_level='UTTERANCE' --gpu=0

# extract textual features
python preprocess.py generate_transcription_files_IEMOCAP
cd feature_extraction/text
python extract_text_embedding_LZ.py --dataset='IEMOCAPFour' --feature_level='UTTERANCE' --model_name='deberta-large' --gpu=0

###################################################################
# We also provide pre-extracted multimodal features
IEMOCAP: https://drive.google.com/file/d/1Hn82-ZD0CNqXQtImd982YHHi-3gIX2G3/view?usp=share_link  -> ./dataset/IEMOCAP/features
CMUMOSI: https://drive.google.com/file/d/1aJxArYfZsA-uLC0sOwIkjl_0ZWxiyPxj/view?usp=share_link  -> ./dataset/CMUMOSI/features
CMUMOSEI:https://drive.google.com/file/d/1L6oDbtpFW2C4MwL5TQsEflY1WHjtv7L5/view?usp=share_link  -> ./dataset/CMUMOSEI/features
~~~~

## CUMMOSI dataset download

## 🚀 Installation

The first step is to download the SDK:

```bash
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git
```

Next, you need to install the SDK on your python enviroment.

```bash
cd CMU-MultimodalSDK
pip install .
```

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yuntaoshou/LMAM&type=Date)](https://star-history.com/#yuntaoshou/LMAM&Date)
