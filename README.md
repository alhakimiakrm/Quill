<h1> Quill: An Exercise in NLP </h1>
I created Quill to venture into NLP and Machine Learning.
 With doing my Dual Degree in English and Computer Science, I thought of no better way to take advantage of what I have been able to learn about the concept
 of a language and the literature behind the history of languages than to incorporate NLP into my studies.



## Prerequisites <a name = "prerequisites"></a>

You need to have a machine with Python 3.9+ installed. Please also refer to the dependencies and ensure you have the required libraries installed.

```Shell

$ python3.9 -V
Python 3.9.7

$ echo $SHELL
/usr/bin/zsh

```

## Development Environment <a name = "Quill"></a>
It is recommended that you initiate a virtual environment to develop out of this repository.

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


<h1> 3-30-2024 </h1> 
 </p>Began training a model on Hemingway's text which have all been consolidated into one file (hemingway1.txt). Also began to do some practice with Word2Vec and playing with positive and negative skip-grams.
</p>
 
 <h2> Quill </h2>
Quill is a poem verse generator that creates poems in the style of various famous literary authors and poets such as Hemingway, Angelou and more. It is (going to be) developed using 
Pyhton and various NLP based technologies like NLTK, Word2Vec and PyTorch amongst any others I might stumble across.

 This exercise is for my own enjoyment and to develop my programming skills as well as other skills such as writing _good_ code as opposed to just working code, proper documentation and good habits when it comes to organizing files, managing a repository, etc etc. If anyone runs into this and wants to help, or provide feedback, you are completely welcome to. 

_This README will constantly be updated as the project moves along. Last update: 1:53 PM EDT // 03/30/2024_
