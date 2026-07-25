from .TextEmbedTransformerModels import *
from .VisionTransformerModels import *


class CLIP(ConfigModule):
    def __init__(self, image_encoder_config: dict = {'image_shape': (3, 512, 512), 'patch_size': 32, 'd_model': 768,
                                                     'head_nums': 12, 'dropout': .1, 'layer_num': 12,
                                                     'mlp_ratio': 2, 'out_dim': 512, "checkpoint_enable": True},
                 text_encoder_config: dict = {'max_seq_length': 128, 'vocab_size': 4096, 'd_model': 512,
                                              'head_nums': 8,
                                              'layer_num': 8, 'dropout': .1,
                                              'mlp_ratio': 2, 'out_dim': 512, "checkpoint_enable": True}):
        super(CLIP, self).__init__()
        assert image_encoder_config["out_dim"] == text_encoder_config["out_dim"], \
            f"文本编码器和图片编码器的输出维度必须相同, {image_encoder_config['out_dim']}!{text_encoder_config['out_dim']}"
        self.image_encoder = VIT(**image_encoder_config)
        self.text_encoder = TextEmbedTransformer(**text_encoder_config)
        self.scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_func = nn.CrossEntropyLoss()
        self.config["image_encoder"] = self.image_encoder.config
        self.config["text_encoder"] = self.text_encoder.config

    def get_model_config(self):
        return {self.image_encoder.__class__.__name__: self.image_encoder.config,
                self.text_encoder.__class__.__name__: self.text_encoder.config}

    def loss(self, per_text, per_image):
        batch_size = per_image.shape[0]
        labels = torch.arange(batch_size, device=per_image.device)
        loss_img = self.loss_func(per_image, labels)
        loss_txt = self.loss_func(per_text, labels)
        total_loss = (loss_img + loss_txt) / 2
        return total_loss, loss_txt.detach().cpu().numpy(), loss_img.detach().cpu().numpy()

    def forward(self, text, images):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(text)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        scale = self.scale.exp()
        per_image = scale * image_features @ text_features.T
        per_text = per_image.T
        loss = self.loss(per_image=per_image, per_text=per_text)
        return loss

    @torch.no_grad()
    def encode_image(self, images):
        image_features = self.image_encoder(images)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        return image_features

    @torch.no_grad()
    def encode_text(self, text):
        text_features = self.text_encoder(text)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features

    @torch.no_grad()
    def text_image_match(self, image, text):
        image_encode = self.encode_image(image)
        text_encode = self.encode_text(text)
        log_scale = self.scale.exp()
        image_pre = (log_scale * image_encode @ text_encode.T).softmax(dim=-1)
        text_pre = image_pre.T
        image_index = image_pre.argmax(-1)
        text_index = text_pre.argmax(-1)
        return text_index, image_index

    @torch.no_grad()
    def recall_at_k(self, image, text, top_k_arr=(1, 5, 10)):
        assert image.shape[0] == text.shape[0], "batch维度不一致"
        batch_size, *_ = image.shape
        image_encode = self.encode_image(image)
        text_encode = self.encode_text(text)
        log_scale = self.scale.exp()
        sim_matrix = (log_scale * image_encode @ text_encode.T).softmax(dim=-1)
        device = sim_matrix.device
        batch_size = sim_matrix.shape[0]
        max_k = max(top_k_arr)
        assert max_k <= batch_size, "样本数小于最大召回数， 计算数值无效"
        labels = torch.arange(batch_size, device=device).reshape(batch_size, 1)
        i2t = sim_matrix
        t2i = sim_matrix.T
        out = []
        for k in top_k_arr:
            i2t_top_k_index = torch.topk(i2t, k=k, dim=-1, largest=True, sorted=True).indices
            t2i_top_k_index = torch.topk(t2i, k=k, dim=-1, largest=True, sorted=True).indices
            i2t_result = (labels == i2t_top_k_index).any(dim=-1)
            t2i_result = (labels == t2i_top_k_index).any(dim=-1)
            i2t_acc = sum(i2t_result) / batch_size
            t2i_acc = sum(t2i_result) / batch_size
            out.append({f"recall@{k}": {"image2text_acc": i2t_acc.item(), "text2image_acc": t2i_acc.item()}})
        return out

    @torch.no_grad()
    def accuracy(self, image, text):
        acc_dict = self.recall_at_k(image=image, text=text, top_k_arr=(1,))
        return acc_dict[0]["recall@1"]["image2text_acc"], acc_dict[0]["recall@1"]["image2text_acc"]
