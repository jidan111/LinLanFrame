from .CLIP import *
from .ImageGenerator import *
from .import_package import *


class Text2Image(object):
    def __init__(self, tokenizer: Tokenizer, clip: CLIP, generator_model: Diffusion, auto_encoder: AutoEncoder,
                 latent_std: float):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = tokenizer
        self.clip = clip.to(self.device)
        self.generator_model = generator_model.to(self.device)
        self.auto_encoder = auto_encoder.to(self.device)
        self.latent_std = latent_std
        self.preprocessing_model(self.clip)
        self.preprocessing_model(self.auto_encoder)
        self.preprocessing_model(self.generator_model)

    def preprocessing_model(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def __call__(self, text, batch_size, fp=None, out_numpy=False):
        if type(text) == str:
            text = [text] * batch_size
        assert len(text) == batch_size
        row = int(math.sqrt(batch_size))
        out = None
        tokens = torch.from_numpy(self.tokenizer(text_array=text)).to(self.device)
        with torch.no_grad(), autocast():
            text_embed = self.clip.encode_text(tokens)
            latent = self.generator_model.sample(batch_size=batch_size, condition=text_embed)
            latent = latent * self.latent_std
            out = self.auto_encoder.latent2image(latent).clamp(min=-1, max=1)
        if fp is not None:
            save_image(normalize=True, padding=1, fp=fp, tensor=out, nrow=row)
        if out_numpy:
            out = make_grid(normalize=True, padding=1, fp=fp, tensor=out, nrow=row)
        return out
