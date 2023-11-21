# MultimodalTransformers



## CLIP

This is a simple implementation of **Natural Language-based Image Search** inspired by the [CLIP](https://openai.com/blog/clip/) approach as proposed by the paper [**Learning Transferable Visual Models From Natural Language Supervision**](https://arxiv.org/abs/2103.00020) by OpenAI in [**PyTorch Lightning**](https://www.pytorchlightning.ai/). We also use [**Weights & Biases**](wandb.ai) for experiment tracking, visualizing results, comparing performance of different backbone models, hyperparameter optimization and to ensure reproducibility.

```shell
python examples/train_clip.py 
```
This command will initialize a CLIP model with a **ResNet50** image backbone and a **distilbert-base-uncased** text backbone. 

## 📚 CLIP: Connecting Text and Images
CLIP (Contrastive Language–Image Pre-training) builds on a large body of work on zero-shot transfer, natural language supervision, and multimodal learning. CLIP pre-trains an image encoder and a text encoder to predict which images were paired with which texts in our dataset. This behavior turns CLIP into a zero-shot classifier. All of a dataset’s classes are converted into captions such as “a photo of a dog” followed by predicting the class of the caption in which CLIP estimates best pairs with a given image.

You can read more about CLIP [here](https://openai.com/blog/clip/) and [here](https://arxiv.org/abs/2103.00020)

## 💿 Dataset
This implementation of CLIP supports training on two datasets [Flickr8k](https://forms.illinois.edu/sec/1713398) which contains ~8K images with 5 captions for each image and [Flickr30k](https://aclanthology.org/Q14-1006/) which contains ~30K images with corresponding captions.

## 🤖 Model
A CLIP model uses a text encoder and an image encoder. This repostiry supports pulling image models from [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) and transformer models from [huggingface transformers](https://github.com/huggingface/transformers). 

