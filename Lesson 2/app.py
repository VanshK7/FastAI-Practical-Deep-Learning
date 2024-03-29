from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()

learn=load_learner('model.pkl')

#im = PILImage.create('dog.jpg')
#learn.predict(im)               # It predicts cat=False, which means dog=True, therefore the prediction is correct

# Converting it into a function
categories=('Dog', 'Cat')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

#classify_image(im)

# Making it Gradio compatible
image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples=['dog.jpg', 'cat.jpg', 'dunno.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)