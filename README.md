This repository is a part of the thesis artifacts. It contains modified codes from [auto-avsr](https://github.com/mpc001/auto_avsr), [torch audio](https://github.com/pytorch/audio), adapted to the new way of preprocessing visual stream. To use any of the scripts, you can follow the same instructions as in the original repositories. To train the Auto-AVSR in the same way as described in the thesis, you'll need to add the [`avsr_trlrwlrs2lrs3vox2avsp_base.pth`](https://drive.google.com/file/d/1mU6MHzXMiq1m6GI-8gqT2zc2bdStuBXu/view?usp=sharing) to the root directory of the project. A more detailed explanation for Auto AVSR is provided below. For the Real-time version - refer to [avsr](./avsr/)

## AUTO AVSR retraining
We changed the Video Preprocessor and corresponding parts of the code to work correctly with the [new visual representation](https://github.com/sasha-tsepilova/lipreading_enhancment/tree/visual_preprocessing). For convenience, the interface is kept the same. For training the model, we used transfer learning techniques, retraining only the visual front, so if you want to run it, you should include [`avsr_trlrwlrs2lrs3vox2avsp_base.pth`](https://drive.google.com/file/d/1mU6MHzXMiq1m6GI-8gqT2zc2bdStuBXu/view?usp=sharing) to the root directory of the project.
For easier usage we provide the related parts from [auto-avsr](https://github.com/mpc001/auto_avsr) readme here:

### Prerequisites
After cloning the repo, you can use the following commands to install prerequisites.
```Shell
pip install torch torchvision torchaudio
pip install pytorch-lightning==1.5.10
pip install sentencepiece
pip install av
pip install hydra-core --upgrade
``` 

You'll also need to install `fairseq` separately:

```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
cd ..
```
The last step is to prepare the dataset. See the instructions in the [preparation](./preparation) folder.

### Training

```Shell
python train.py exp_dir=[exp_dir] \
               exp_name=[exp_name] \
               data.modality=[modality] \
               data.dataset.root_dir=[root_dir] \
               data.dataset.train_file=[train_file] \
               trainer.num_nodes=[num_nodes] \
```
<details open>
  <summary><strong>Required arguments</strong></summary>

- `exp_dir`: Directory to save checkpoints and logs to.
- `exp_name`: Experiment name. Location of checkpoints is `[exp_dir]`/`[exp_name]`.
- `data.modality`: Type of input modality, valid values: `video` and `audio`.
- `data.dataset.root_dir`: Root directory of preprocessed dataset, default: `null`.
- `data.dataset.train_file`: Filename of training label list, default: `lrs3_train_transcript_lengths_seg24s.csv`.
- `trainer.num_nodes`: Number of machines used, default: 1.
- `trainer.resume_from_checkpoint`: Path of the checkpoint from which training is resumed, default: `null`.

</details>

<details>
  <summary><strong>Optional arguments</strong></summary>

- `data.dataset.val_file`: Filename of validation label list, default: `lrs3_test_transcript_lengths_seg24s.csv`.
- `pretrained_model_path`: Path to the pre-trained model, default: `null`.
- `transfer_frontend` Flag to load the weights of the front-end module works with `pretrained_model_path.`
- `transfer_encoder` Flag to load the weights of encoder, works with `pretrained_model_path`.
- `trainer.max_epochs`: Number of epochs, default: 75.
- `trainer.gpus`: Number of GPUs to train on on each machine, default: -1, which use all gpus.
- `data.max_frames`: Maximal number of frames in a batch, default: 1800.
- `optimizer.lr`: Learning rate, default: 0.001.

</details>


<details open>
  <summary><strong>Note</strong></summary>

- For lrs3, start by training from scratch on a subset (23h, max duration=4 seconds) at a learning rate of 0.0002 (see [model-zoo](#model-zoo)). Then, fine-tune on the full set with a learning rate of 0.001. A script for subset creation is available [here](./preparation/limit_length.py). Please refer to [instruction](INSTRUCTION.md) for training new datasets.
- If you want to monitor the training process, customize [logger](https://lightning.ai/docs/pytorch/1.5.8/api_references.html#loggers-api) within `pytorch_lightning.Trainer()`.
- To maximize resource utilization, set `data.max_frames` to the largest to fit into your GPU memory.

</details>

### Testing

```Shell
python eval.py data.modality=[modality] \
               data.dataset.root_dir=[root_dir] \
               data.dataset.test_file=[test_file] \
               pretrained_model_path=[pretrained_model_path] \
```

<details open>
  <summary><strong>Required arguments</strong></summary>

- `data.modality`: Type of input modality, valid values: `video`, `audio` and `audiovisual`.
- `data.dataset.root_dir`: Root directory of preprocessed dataset, default: `null`.
- `data.dataset.test_file`: Filename of testing label list, default: `lrs3_test_transcript_lengths_seg24s.csv`.
- `pretrained_model_path`: Path to the pre-trained model, set to `[exp_dir]/[exp_name]/model_avg_10.pth`, default: `null`.

</details>

<details>
  <summary><strong>Optional arguments</strong></summary>

- `decode.snr_target=[snr_target]`: Level of signal-to-noise ratio (SNR), default: 999999.

</details>

### Demo

Want to see how our asr/vsr model performs on your audio/video? Just run this command:

```Shell
python demo.py  data.modality=[modality] \
                pretrained_model_path=[pretrained_model_path] \
                file_path=[file_path]
```
<details open>
  <summary><strong>Required arguments</strong></summary>

- `data.modality`: Type of input modality, valid values: `video` and `audio`.
- `pretrained_model_path`: Path to the pre-trained model.
- `file_path`: Path to the file for testing.

</details>

## Results

Model | Training data (training hours) | WER on LRS3 test
--- | --- | --- 
Auto AVSR | LRS3 + VOXCeleb (3448) | 0.93
Auto AVSR | LRW (157) + LRS 3 (438) | 2.3
Ours | 25% LRS3 (110) | 4.2
RNN-T | YT (31000) | 4.5

Due to the lack of resources, we couldn't use more training data, but this table shows that we can achieve good results with the new preprocessing method, even on a small dataset.


## Acknowledgement

This repository is built using the [auto-avsr](https://github.com/mpc001/auto_avsr), [torch audio](https://github.com/pytorch/audio) (specifically avsr part) repositories.


