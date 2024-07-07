
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

I've implemented an LSTM model to train on Hemingway's limited works. As of the last commit of this README file, only 20 epochs were used to train on the corpus. Here is an example of the output generated so far:

### Test start text: 'soldiers never do die well'

```
soldiers never do die well crosses mark the places wooden crosses where they fell stuck above their faces soldiers pitch and cough and twitch all the world roars red and black soldiers smother in a ditch choking through the whole attack i like americans they are so unlike canadians they do not take their policemen
```

### Test start text: 'the age demanded'

```
the age demanded that we sing and cut away our tongue the age demanded that we flow and hammered in the bung the age demanded that we dance and jammed us into iron pants and in the end the age was handed the sort of shit that it demanded a porcupine skin stiff
```

Very poetic, isn't it?

Raising the number of epochs could lead to overfitting, so an important implementation to work on would be early stopping. Early stopping halts the training process if the model's performance on a validation set stops improving for a certain number of epochs. Another useful technique to consider is regularization; using methods like L1/L2 regularization can help prevent the model from overfitting the data.

_Last update: 4:22 PM EDT // 7/7/2024_
```
