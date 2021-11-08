import os
from transformers import pipeline

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

classifier = pipeline("sentiment-analysis")
print(classifier("I've been waiting for a HuggingFace course my whole life."))