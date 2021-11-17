import os
from transformers import pipeline

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)

# sentence --> words --> tokens

from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="tf")
print(inputs)

# token --> logits

from transformers import TFAutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = TFAutoModel.from_pretrained(checkpoint)

outputs = model(inputs)
print(outputs.last_hidden_state.shape)

from transformers import TFAutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(inputs)

print(outputs.logits.shape)

print(outputs.logits)

# logits to predictions

import tensorflow as tf

predictions = tf.math.softmax(outputs.logits, axis=-1)
print(predictions)

model.config.id2label