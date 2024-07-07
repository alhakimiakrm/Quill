<h1> Quill: An Exercise in NLP </h1>
I created Quill to venture into NLP and Machine Learning. This program takes a corpus of texts from pre-determined authors,
process/cleans up their work, and output a new poem or short story based on the training data gathered from their work. 



## Prerequisites <a name = "prerequisites"></a>

You need to have a machine with Python 3.9+ installed. Please also refer to requirements and run 'pip install -r requirements.txt'
```Shell

$ python3.9 -V
Python 3.9.7

$ echo $SHELL
/usr/bin/zsh

```

## Development Environment <a name = "Quill"></a>
It is recommended to create and activate a virtual environment before contributing to this repository.

```Shell
$ pip install virtualenv
```

On PC:

```Shell
$ python -m venv <your-environment-name>
$ <\.your-enviornment-name>\Scripts\activate
```

On Mac/Linux:
```Shell
$ python3 venv <your-environment-name>
$ source <your-environment-name>/bin/activate
```

Use deactivate to exit the virtual environment
```Shell
$ deactivate
```
 
 <h2> Quill </h2>
I've implemented an LSTMM model to train on Hemingway's short handful of work. As of the last commit of this readme file, only 20 epochs were put in place to train on the work. Here is an example of what is being output so far.

<h5> Test start text: 'soldiers never do die well'

>soldiers never do die well crosses mark the places wooden crosses where they fell stuck above their faces soldiers pitch and cough and twitch all the world roars red and black soldiers smother in a ditch choking through the whole attack i like americans they are so unlike canadians they do not take their policemen

<h5> Test start text: 'the age demanded'

>the age demanded that we sing and cut away our tongue the age demanded that we flow and hammered in the bung the age demanded that we dance and jammed us into iron pants and in the end the age was handed the sort of shit that it demanded a porcupine skin stiff

Very poetic, isn't it.

Raising the number of epochs would lead to overfitting so an important implementation to work on would be early stopping, where the training process is halted if the model's performance on a validation sets stops improving for a certain number of epochs. Another good thing to 
look into would be regularization; using techniques like L1/L2 regularization might prevent the model from overfitting its data.

_Last update: 4:22 PM EDT // 7/7/2024_
