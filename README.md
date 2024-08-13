
# Quill: An Exercise in NLP

I created Quill to venture into NLP and Machine Learning. This program takes a corpus of texts from pre-determined authors, processes/cleans their work, and outputs a new poem or short story based on the training data gathered from their work.

## Prerequisites <a name="prerequisites"></a>

You need to have a machine with Python 3.9+ installed. Please also refer to the requirements and run the following command to install them:

```shell
pip install -r requirements.txt
```

To check your Python version and shell, use:

```shell
$ python3.9 -V
Python 3.9.7

$ echo $SHELL
/usr/bin/zsh
```

## Development Environment <a name="Quill"></a>

It is recommended to create and activate a virtual environment before contributing to this repository.

Install `virtualenv`:

```shell
$ pip install virtualenv
```

### On Windows:

```shell
$ python -m venv <your-environment-name>
$ <your-environment-name>\Scripts\activate
```

### On Mac/Linux:

```shell
$ python3 -m venv <your-environment-name>
$ source <your-environment-name>/bin/activate
```

To deactivate the virtual environment:

```shell
$ deactivate
```

## Quill

I've implemented an LSTM model to train on Hemingway and Frost's work. The model and training parameters are in accordance to MY personal computer spec's below, so please proceed with caution when developing out of your own computer.

### Specs:

```S: Pop!_OS 22.04 LTS x86_64       
Shell: zsh 5.8.1  
Resolution: 2560x1440  
DE: GNOME 42.9  
WM: Mutter  
WM Theme: Nordic  
Terminal: kitty  
CPU: AMD Ryzen 7 5800X (16) @ 4.375GHz  
GPU: RTX 4080 (NVIDIA CORPORATION)
Memory: 4887MiB / 64227MiB
```

Raising the number of epochs could lead to overfitting as well, so if you're wondering what you could contribute- an important implementation to work on would be early stopping. Early stopping halts the training process if the model's performance on a validation set stops improving for a certain number of epochs. Another useful technique to consider is regularization; using methods like L1/L2 regularization can help prevent the model from overfitting the data.

_Last update: 4:05 PM EDT // 8/13/2024_
