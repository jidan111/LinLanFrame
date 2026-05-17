from .TextEmbedTransformerModels import *
from .VisionTransformerModels import *


class CLIP(ConfigModule):
    def __init__(self, max_seq_length=128, image_shape=(3, 64, 64), vocab_size=2048, d_model=512, head_num=8,
                 hidden_dim=512, patch_size=16, layer_num=8, dropout=.1):
        super(CLIP, self).__init__()
        self.image_encoder = VIT(image_shape=image_shape, patch_size=patch_size, d_model=d_model, head_num=head_num,
                                 dropout=dropout,
                                 pool_type="mean", layer_num=layer_num, hidden_dim=hidden_dim)
        self.text_encoder = TextEmbedTransformer(max_seq_length=max_seq_length, vocab_size=vocab_size, d_model=d_model,
                                                 head_num=head_num,
                                                 layer_num=layer_num, dropout=dropout,
                                                 hidden_dim=hidden_dim)
        self.scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = nn.CrossEntropyLoss()

    def get_model_config(self):
        return {self.image_encoder.__class__.__name__: self.image_encoder.config,
                self.text_encoder.__class__.__name__: self.text_encoder.config}

    def loss(self, per_text, per_image):
        batch_size = per_image.shape[0]
        # 标签：对角线是正样本（第i张图对应第i句文本）
        labels = torch.arange(batch_size, device=per_image.device)
        # 1. 图像→文本 损失
        loss_img = self.loss_func(per_image, labels)
        # 2. 文本→图像 损失
        loss_txt = self.loss_func(per_text, labels)
        # 3. 总损失：取平均（对称损失）
        total_loss = (loss_img + loss_txt) / 2
        return total_loss, loss_txt.detach().cpu().numpy(), loss_img.detach().cpu().numpy()

    def forward(self, text, images):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(text)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        scale = self.scale.exp()
        per_image = scale * image_features @ text_features.t()
        per_text = per_image.t()
        # return per_image, per_text
        loss = self.loss(per_image=per_image, per_text=per_text)
        return loss

    @torch.no_grad()
    def text_embed(self, tokens):
        return self.text_encoder.encode_text(tokens)

    @torch.no_grad()
    def encode_image(self, images):
        image_features = self.image_encoder(images)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    @torch.no_grad()
    def encode_text(self, text):
        text_encode = self.text_encoder(text)
        text_encode = text_encode / text_encode.norm(dim=1, keepdim=True)
        return text_encode

    @torch.no_grad()
    def image_text_match(self, image, text):
        image_encode = self.encode_image(image)
        text_encode = self.encode_text(text)
        log_scale = self.scale.exp()
        image_pre = log_scale * torch.matmul(image_encode, text_encode.t())
        text_pre = image_pre.t()
        image_index = image_pre.argmax(-1)
        text_index = text_pre.argmax(-1)
        return image_index, text_index

    @torch.no_grad()
    def accuracy(self, image, text):
        assert image.shape[0] == text.shape[0], "batch维度不一致"
        batch_size, *_ = image.shape
        image_encode = self.encode_image(image)
        text_encode = self.encode_text(text)
        log_scale = self.scale.exp()
        image_pre = log_scale * torch.matmul(image_encode, text_encode.t())
        text_pre = image_pre.t()
        image_index = image_pre.argmax(-1)
        text_index = text_pre.argmax(-1)
        labels = torch.arange(text_pre.shape[0]).to(text_pre.device)
        image_cnt = torch.sum(image_index == labels)
        text_cnt = torch.sum(text_index == labels)
        image_acc = image_cnt / batch_size
        text_acc = text_cnt / batch_size
        # print("image2text准确率:{:.2%}, text2image准确率:{:.2%}".format(image_acc, text_acc))
        return image_acc, text_acc
