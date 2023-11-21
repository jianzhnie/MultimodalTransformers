import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from transformers import AutoModel, BertConfig


class ImageEncoder(nn.Module):
    """Encode images to a fixed size vector."""

    def __init__(
        self,
        model_name: str = "resnet50",
        pretrained: bool = True,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        # self.model = timm.create_model(
        #     model_name, pretrained, num_classes=0, global_pool="avg"
        # )
        self.model = torchvision.models.resnet50(weights="DEFAULT")
        self.model.fc = nn.Identity()

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x) -> torch.Tensor:
        output = self.model(x)
        return output


class TextEncoder(nn.Module):
    """"""

    def __init__(
        self, model_name: str, pretrained: bool = True, trainable: bool = True
    ) -> None:
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            config = BertConfig()
            self.model = AutoModel.from_config(config)

        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        projection_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        output = self.gelu(projected)
        output = self.fc(output)
        output = self.dropout(output)
        output += projected
        output = self.layer_norm(output)
        return output


class CLIPModel(nn.Module):
    def __init__(
        self,
        image_encoder_alias: str,
        text_encoder_alias: str,
        image_encoder_pretrained: bool = True,
        text_encoder_pretrained: bool = True,
        image_encoder_trainable: bool = True,
        text_encoder_trainable: bool = True,
        image_embedding_dims: int = 2048,
        text_embedding_dims: int = 768,
        projection_dims: int = 256,
        dropout: float = 0.0,
        temperature: float = 1.0,
        logit_scale_init_value: float = 1.0,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(
            model_name=image_encoder_alias,
            pretrained=image_encoder_pretrained,
            trainable=image_encoder_trainable,
        )
        self.text_encoder = TextEncoder(
            model_name=text_encoder_alias,
            pretrained=text_encoder_pretrained,
            trainable=text_encoder_trainable,
        )
        self.image_projection = ProjectionHead(
            embedding_dim=image_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.text_projection = ProjectionHead(
            embedding_dim=text_embedding_dims,
            projection_dim=projection_dims,
            dropout=dropout,
        )
        self.temperature = temperature
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        caption: str,
    ):
        # Getting Image and Text Features
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # normalized features
        image_embeds = image_embeddings / image_embeddings.norm(
            p=2, dim=-1, keepdim=True
        )
        text_embeds = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        # labels
        targets = torch.arange(len(logits), device=logits.device)
        logits_per_text, logits_per_image = logits, logits.t()

        # clip loss
        caption_loss = nn.functional.cross_entropy(
            logits_per_text, targets
        )
        image_loss = nn.functional.cross_entropy(
            logits_per_image, targets
        )
        loss = (caption_loss + image_loss) / 2.0

        output = (logits_per_image, logits_per_text, text_embeds, image_embeds)

        return (loss,) + output

    def forward_custom(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        images: torch.Tensor,
        caption: str,
    ):
        # Getting Image and Text Features
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature,
            dim=-1,
        )
        texts_loss = self.computer_loss(logits, targets, reduction="none")
        images_loss = self.computer_loss(logits.T, targets.T, reduction="none")
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()

    def computer_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, reduction: str = "none"
    ):
        loss = (-targets * self.log_softmax(logits)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
