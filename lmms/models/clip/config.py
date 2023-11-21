import torch


class ClipConfig:
    debug = True
    image_path = '/home/robin/datasets/vision_data/flickr/flickr8k'
    captions_path = '/home/robin/datasets/vision_data/flickr/flickr8k'

    image_model_name = 'resnet50'
    text_model_name = 'bert-base-uncased'
    text_tokenizer_name = 'bert-base-uncased'
    image_embedding_dims = 2048
    text_embedding_dims = 768
    image_encoder_pretrained = True
    text_encoder_pretrained = True
    image_encoder_trainable = False
    text_encoder_trainable = False
    # for projection head; used for both image and text encoders
    projection_dims = 256
    dropout = 0.1

    temperature = 1.0
    # image size
    target_size = 224
    max_length = 256
    # data loader
    batch_size = 64
    num_workers = 4
    # optim
    lr = 1e-3
    weight_decay = 1e-4
    epochs = 10
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
