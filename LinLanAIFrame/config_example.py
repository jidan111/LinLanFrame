GAN_config = {
    "generator": {"in_dim": 128, "image_shape": (3, 64, 64), "hidden_channels": 8, "depth": None, "attention": [],
                  "head_num": 8,
                  "dropout": .1, "resnet_num": 1, "up_mode": "interpolate"},
    "dis": {"image_shape": (3, 64, 64), "hidden_channels": 8, "depth": None}}
Diffusion_config = {"Diffusion": {"model": None, "image_shape": (3, 28, 28), "step_nums": 100, "step_dim": 128,
                                  "schedule_name": "linear", "betas": (1e-4, 0.02)},
                    "UnetConditional": {'image_shape': (3, 64, 64), 'hidden_channels': 8, 'depth': None,
                                        'attention': [],
                                        'd_model': 512, 'head_num': 8,
                                        'dropout': .1, 'resnet_num': 1, 'up_mode': "interpolate", 'condition_dim': 128,
                                        'key_dim': None},
                    "DIT": {'image_shape': (3, 64, 64), 'patch_size': 8, 'condition_dim': 128, 'd_model': 512,
                            'head_nums': 8, 'layer_num': 8, 'key_dim': None,
                            'dropout': .1}}
AutoEncoder_config = {
    "AutoEncoder": {'image_shape': (3, 64, 64), 'depth': 4, 'hidden_channels': 8, 'latent_dim': 4, 'attention': [],
                    'dropout': .1, 'resnet_num': 1,
                    'up_mode': "interpolate", 'head_num': 8},
    "VQAutoEncoder": {'image_shape': (3, 256, 256), 'depth': 2, 'hidden_channels': 64, 'latent_dim': 4, 'attention': [],
                      'dropout': .1,
                      'resnet_num': 2, 'up_mode': "ConvTranspose2d", 'head_num': 8, 'beta': 0.25, 'embed_num': 16384,
                      'use_vq_bridge': False,
                      'vq_bridge_patch_size': 64, 'vq_bridge_head_num': 4, 'vq_bridge_layer_num': 2,
                      'vq_bridge_dropout': .1}}
CLIP_config = {"max_seq_length": 80, "image_shape": [3, 256, 256], "vocab_size": 2048, "d_model": 512, "head_num": 8,
               "hidden_dim": 756, "patch_size": 16, "layer_num": 8, "dropout": 0.1}
trainer_config = {
    "CLIPTrainer": {'model': None, 'lr': 1e-4, 'valid_dir': "./valid/", 'save_model_dir': "./model/clip",
                    'compile_model': False,
                    'mid_save_step': 500},
    "DiffusionTrainer": {
        'model': None, 'lr': 1e-5, 'valid_dir': "./valid", 'save_model_dir': "./model/diffusion",
        'compile_model': False, 'mid_save_step': 500
    },
    "GANTrainer": {
        'generator': None, 'discriminator': None, 'gen_lr': 0.0002, 'dis_lr': 0.0002, 'n_critic': 1,
        'valid_dir': "./valid", 'save_model_dir': "./model/gan", 'loss_type': "hinge", 'lambda_gp': 10,
        'mid_save_step': 500
    },
    "AutoEncoderTrainer": {
        'model': None, 'lr': 5e-6, 'valid_dir': "./valid", 'save_model_dir': "./model/autoencoder",
        'perception_net': "vgg", 'kl_weight': 1e-6, 'perception_weight': 1.,
        'valid_batch_size': 9, 'have_perception': False, 'compile_model': False, 'mid_save_step': 500,
        'book_weight': 1.,
        'vae_mode': "va"
    },
    "AutoEncoderWithDiscriminatorTrainer": {
        'model': None, 'discriminator': None, 'gen_lr': 5e-5, 'dis_lr': 1e-4, 'n_critic': 1,
        'valid_dir': "./valid", 'save_model_dir': "./model/gan", 'perception_net': "vgg", 'perception_weight': .3,
        'valid_batch_size': 9, 'have_perception': False, 'mid_save_step': 500, 'kl_weight': 1e-6, 'book_weight': 1.,
        'vae_mode': "va"
    }
}
