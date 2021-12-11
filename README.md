# Simple CNN attribution method
You can see demo online [here](https://share.streamlit.io/485294-muni/streamlit-cnn-vis/main/app.py). The source repository is on [github](https://github.com/485294-muni/streamlit-cnn-vis).

## Background
Neural networks are very black-boxish by nature. This makes them powerful, but also hard to debug. It might be benefitial to visualize what is the basis of their decisions. 

## Idea
We first take an image and let a convolutional neural network classify it.
Then, we then try to find the which pixels currently influence the prediction
confidence the most. Roughly speaking, this can be done by taking the first
derivatives (gradient) of the category prediction w.r.t. image pixels.

There are some limitations to this method. If the network is very confident
about it's prediction, it becomes insensitive to any change in an **individual
pixel**. There are several methods that solve this problem, but they are more
complex and computationally infeasible for an interactive demo app.

I have also included an option to blend (multiply) the original image with
the "explanation mask". While this can create visually interesting output,
and give you a slightly different view on the explanation, keep in mind
that it will inevitable bias the explanation towards bright regions of the
original image.

## Implementation
The project is implemented as a streamlit app. It runs a pretrained
efficientnet convolutional neural network trained on Imagenet dataset.
Note that Imagenet dataset does not contain a category for people, so
the model will not recognize a person.

# Installation
You can run the app by installing the dependencies specified in `requirements.text`,
and then executing `streamlit run app.py` in shell.

Running the app for the first time can take some time because it downloads
the NN weights from the internet (it is cached for subsequent runs though).


## Project structure
```
- app.py
    = main entry point
- requirements.txt
    = project dependencies
- toy.ipynb
    = auxiliary notebook for experimenting and playing around
- classes.txt
    = textual names for categories in imagenet dataset
- style.css
    = css injected into streamit app
- images/
    contains several example files
``` 