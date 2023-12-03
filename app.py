import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('fastapi_model.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = "Chest X-Ray Classifier: Tuberculosis"
description = "An AI app for classification of chest X-Rays as Tuberculosis positive or normal"
article="<p style='text-align: center'><a href='https://tmabraham.github.io/blog/gradio_hf_spaces_tutorial' target='_blank'>Blog post</a></p>"
#examples = ['siamese.jpg']
interpretation='default'
enable_queue=True
gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=2),title=title,description=description,
             #article=article,
             #examples=examples,
             interpretation=interpretation,enable_queue=enable_queue).launch()
