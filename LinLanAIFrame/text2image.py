from .CLIP import *
from .ImageGenerator import *


class Text2Image(object):
    def __init__(self, clip_model: CLIP, bpe_model: BPE, diffusion_model: Diffusion, vae_model: AutoEncoder):
        super(Text2Image, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_seq_dim = clip_model.config["d_model"]
        diffusion_key_dim = diffusion_model.model.config["key_dim"]
        assert clip_seq_dim == diffusion_key_dim, f"文本嵌入维度与扩散模型文本维度不匹配, {clip_seq_dim}!={diffusion_key_dim}"
        self.max_seq_length = clip_model.config["max_seq_length"]
        self.clip_model = clip_model
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.bpe_model = bpe_model
        self.diffusion_model = diffusion_model
        for param in self.diffusion_model.parameters():
            param.requires_grad = False
        self.vae_model = vae_model
        for param in self.vae_model.parameters():
            param.requires_grad = False
        self.vae_model = self.vae_model.to(self.device)
        self.diffusion_model = self.diffusion_model.to(self.device)
        self.clip_model = self.clip_model.to(self.device)

    def __call__(self, text, batch_size=4):
        caption = [text]
        caption = self.bpe_model.encode_sentences(caption, dim=self.max_seq_length, numpy=True)
        caption = torch.tensor(caption, dtype=torch.long, device=self.device)
        caption_embed = self.clip_model.text_embed(caption)
        caption_embed = caption_embed.repeat(batch_size, 1, 1)
        image = self.diffusion_model.sample(batch_size=batch_size, txt=caption_embed, mode="dpm")
        image = self.vae_model.latent2image(image)
        return image
