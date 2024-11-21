# LMAM
![Supported Python versions](https://img.shields.io/badge/%20python-3.8-blue)
![Supported OS](https://img.shields.io/badge/%20Supported_OS-Windows-red)


# D2CHGNN: Dynamic Dual Contrastive Hyperbolic Hypergraph for Incomplete Multi-view Emotion Representation Learning

Correspondence to: 
  - Yuntao Shou (shouyuntao@stu.xjtu.edu.cn)

## Paper
[**GCNet: Graph Completion Network for Incomplete Multimodal Learning in Conversation**](https://arxiv.org/abs/2203.02177)<br>
Zheng Lian, Lan Chen, Licai Sun, Bin Liu, Jianhua Tao<br>
IEEE Transactions on pattern analysis and machine intelligence, 2022

Please cite our paper if you find our work useful for your research:

```tex
@article{lian2022gcnet,
  title={GCNet: Graph Completion Network for Incomplete Multimodal Learning in Conversation},
  author={Lian, Zheng and Chen, Lan and Sun, Licai and Liu, Bin and Tao, Jianhua},
  journal={IEEE Transactions on pattern analysis and machine intelligence},
  year={2022},
  publisher={IEEE}
}
```

## Usage (Choose IEMOCAP-Six for Example)

### Prerequisites
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



### Datasets

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
