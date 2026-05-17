from .structs import *


class AutoEncoder(ConfigModule):
    def __init__(self, image_shape, depth, hidden_channels, latent_dim, attention=[], dropout=.1, resnet_num=1,
                 up_mode="ConvTranspose2d", head_num=8):
        super(AutoEncoder, self).__init__()
        assert (image_shape[1] // (2 ** depth)) * (2 ** depth) == image_shape[1], "深度无法还原为原图片"
        self.depth = depth
        if type(up_mode) == str:
            up_mode = [up_mode] * depth
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.dim = image_shape[1] // (2 ** depth)
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=hidden_channels, kernel_size=1)
        self.encoder = Encoder(depth=depth, hidden_channels=hidden_channels, resnet_num=resnet_num, attention=attention,
                               dropout=dropout, head_num=head_num)
        self.decoder = Decoder(depth=depth, hidden_channels=hidden_channels, resnet_num=resnet_num, mode=up_mode,
                               attention=attention[::-1], dropout=dropout, head_num=head_num)
        self.q_enc = nn.Conv2d(in_channels=self.hidden_channels * (2 ** self.depth),
                               out_channels=latent_dim * 2, kernel_size=1)
        self.p_dec = nn.Conv2d(in_channels=latent_dim, out_channels=self.hidden_channels * (2 ** self.depth),
                               kernel_size=1)
        self.out = nn.Conv2d(in_channels=hidden_channels, out_channels=image_shape[0], kernel_size=1)

    def forward(self, x):
        x = self.init(x)
        x = self.encoder(x)
        q_prior = DiagonalGaussianDistribution(self.q_enc(x))
        z = q_prior.sample()
        z = self.p_dec(z)
        z = self.decoder(z)
        out = self.out(z)
        return F.tanh(out), q_prior

    def get_last_layer_weight(self):
        return self.out.weight

    @torch.no_grad()
    def image2featrues(self, x):
        x = self.init(x)
        x = self.encoder(x)
        x = self.q_enc(x)
        return x

    @torch.no_grad()
    def image2latent(self, x):
        x = self.init(x)
        x = self.encoder(x)
        x = DiagonalGaussianDistribution(self.q_enc(x))
        return x.sample()

    @torch.no_grad()
    def latent2image(self, x):
        z = self.p_dec(x)
        z = self.decoder(z)
        out = self.out(z)
        return F.tanh(out)

    @torch.no_grad()
    def sample(self, batch_size):
        device = self.init.weight.device
        noise = torch.randn(size=(batch_size, self.latent_dim, self.dim, self.dim), device=device)
        out = self.latent2image(noise)
        return out


class VQBridgeProjectorAttention(nn.Module):
    def __init__(self, embed_num, embed_dim, head_num=4, layer_num=2, dropout=.1, mul_size=64):
        super(VQBridgeProjectorAttention, self).__init__()
        self.d_model = embed_dim * mul_size
        self.embed_dim = embed_dim
        self.embed_num = embed_num
        self.init = nn.Linear(embed_dim, self.d_model)
        self.out = nn.Linear(self.d_model, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(size=(1, embed_num, embed_dim)))
        self.vit = TransformerEncoder(query_dim=self.d_model, d_model=self.d_model,
                                      head_num=head_num,
                                      layer_num=layer_num,
                                      dropout=dropout, hidden_dim=self.d_model)

    def forward(self, codebook):
        codebook = codebook.unsqueeze(0)  # [1, N, D]
        codebook = codebook + self.pos_embed  # [1, N, D]
        codebook = self.init(codebook)
        codebook = self.vit(codebook)  # [1, N, D]
        codebook = self.out(codebook)
        codebook = codebook.squeeze(0)  # [N, D]
        return codebook


class VectorQuantizer(nn.Module):
    def __init__(self, embed_num, embed_dim, beta=0.25, latent_shape=(4, 64, 64),
                 use_vq_bridge=True, vq_bridge_mul_size=64, vq_bridge_head_num=4, vq_bridge_layer_num=2,
                 vq_bridge_dropout=.1):
        super(VectorQuantizer, self).__init__()
        self.embed_num = embed_num
        self.embed_dim = embed_dim
        self.beta = beta
        self.use_vq_bridge = use_vq_bridge
        self.latent_shape = (latent_shape[1], latent_shape[2], latent_shape[0])
        self.embedding = nn.Embedding(num_embeddings=self.embed_num, embedding_dim=self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.embed_num, 1.0 / self.embed_num)
        # ===== VQBridge核心改造点 =====
        if use_vq_bridge:
            self.vq_bridge = VQBridgeProjectorAttention(
                embed_num=embed_num,
                embed_dim=embed_dim,
                head_num=vq_bridge_head_num,
                layer_num=vq_bridge_layer_num,
                dropout=vq_bridge_dropout,
                mul_size=vq_bridge_mul_size
            )

    def forward(self, z):
        optimized_codebook = self.embedding.weight
        if self.use_vq_bridge:
            optimized_codebook = self.vq_bridge(optimized_codebook)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embed_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(optimized_codebook ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, optimized_codebook.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = optimized_codebook[min_encoding_indices].reshape(z.shape)
        # ===== 3. 损失函数改造 =====
        # 1) Commitment loss：约束编码器向优化码本靠拢
        commitment_loss = torch.mean((z_q.detach() - z) ** 2)
        # 2) Codebook loss：约束优化码本向编码器输出靠拢
        codebook_loss = self.beta * torch.mean((z_q - z.detach()) ** 2)
        # 3) VQBridge正则化：防止优化码本偏离原始分布
        # vq_bridge_reg = 0.01 * F.mse_loss(optimized_codebook, self.embedding.weight)
        # total_loss = commitment_loss + codebook_loss + vq_bridge_reg
        total_loss = commitment_loss + codebook_loss
        # STE直通估计（保持离散特性）
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, total_loss, min_encoding_indices

    def get_codebook_entry(self, indices, shape=None, optimized_codebook=None):
        if optimized_codebook is None:
            if self.use_vq_bridge:
                optimized_codebook = self.vq_bridge(self.embedding.weight)
            else:
                optimized_codebook = self.embedding.weight
        z_q = optimized_codebook[indices]
        if not shape:
            z_q = z_q.view(-1, *self.latent_shape)
        else:
            z_q = z_q.view(shape)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q

    @torch.no_grad()
    def feature2latent(self, z):
        batch_size, c, h, w = z.shape
        optimized_codebook = self.embedding.weight
        if self.use_vq_bridge:
            optimized_codebook = self.vq_bridge(optimized_codebook)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embed_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(optimized_codebook ** 2, dim=1) - \
            2 * torch.matmul(z_flattened, optimized_codebook.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        latent = self.get_codebook_entry(indices=min_encoding_indices, shape=(batch_size, h, w, c),
                                         optimized_codebook=optimized_codebook)
        return latent


class VQAutoEncoder(ConfigModule):
    def __init__(self, image_shape, depth, hidden_channels, latent_dim, attention=[], dropout=.1, resnet_num=1,
                 up_mode="ConvTranspose2d", head_num=8, beta=0.25, embed_num=1024, use_vq_bridge=True,
                 vq_bridge_mul_size=64, vq_bridge_head_num=4, vq_bridge_layer_num=2,
                 vq_bridge_dropout=.1):
        super(VQAutoEncoder, self).__init__()
        assert (image_shape[1] // (2 ** depth)) * (2 ** depth) == image_shape[1], "深度无法还原为原图片"
        self.depth = depth
        if type(up_mode) == str:
            up_mode = [up_mode] * depth
        self.image_shape = image_shape
        self.embed_num = embed_num
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.dim = image_shape[1] // (2 ** depth)
        self.init = nn.Conv2d(in_channels=image_shape[0], out_channels=hidden_channels, kernel_size=1)
        self.encoder = Encoder(depth=depth, hidden_channels=hidden_channels, resnet_num=resnet_num, attention=attention,
                               dropout=dropout, head_num=head_num)
        self.decoder = Decoder(depth=depth, hidden_channels=hidden_channels, resnet_num=resnet_num, mode=up_mode,
                               attention=attention[::-1], dropout=dropout, head_num=head_num)
        self.q_enc = nn.Conv2d(in_channels=self.hidden_channels * (2 ** self.depth),
                               out_channels=latent_dim, kernel_size=1)
        self.p_dec = nn.Conv2d(in_channels=latent_dim, out_channels=self.hidden_channels * (2 ** self.depth),
                               kernel_size=1)
        self.book = VectorQuantizer(embed_num=embed_num, embed_dim=latent_dim, beta=beta,
                                    latent_shape=(latent_dim, self.dim, self.dim), use_vq_bridge=use_vq_bridge,
                                    vq_bridge_mul_size=vq_bridge_mul_size, vq_bridge_head_num=vq_bridge_head_num,
                                    vq_bridge_layer_num=vq_bridge_layer_num,
                                    vq_bridge_dropout=vq_bridge_dropout)
        self.out = nn.Conv2d(in_channels=hidden_channels, out_channels=image_shape[0], kernel_size=1)

    def forward(self, x):
        x = self.init(x)
        encode = self.encoder(x)
        encode = self.q_enc(encode)
        z, book_loss, index = self.book(encode)
        out = self.p_dec(z)
        out = self.decoder(out)
        out = self.out(out)
        return F.tanh(out), book_loss

    def get_last_layer_weight(self):
        return self.out.weight

    @torch.no_grad()
    def image2latent(self, x):
        x = self.init(x)
        encode = self.encoder(x)
        encode = self.q_enc(encode)
        latent = self.book.feature2latent(encode)
        return latent

    @torch.no_grad()
    def latent2image(self, x):
        out = self.p_dec(x)
        out = self.decoder(out)
        out = self.out(out)
        return F.tanh(out)

    @torch.no_grad()
    def sample(self, batch_size):
        device = self.init.weight.device
        noise = torch.randint(low=0, high=self.embed_num, size=(batch_size * self.dim * self.dim),
                              dtype=torch.long, device=device)
        out = self.book.get_codebook_entry(noise, shape=(batch_size, self.dim, self.dim, self.latent_dim))
        out = self.latent2image(out)
        return out
