import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    """
    def __init__(self, in_channels:int=3, patch_size:int=16, embedding_dim:int=768):
        super().__init__()
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.patch_size = patch_size
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisble by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)

        return x_flattened.permute(0, 2, 1) 

class MSABlock(nn.Module):
    def __init__(self, embedding_dim:int=768,  num_heads:int=12,  attn_dropout:float=0): 
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True) 
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, need_weights=False) 
        return attn_output

class MLPBlock(nn.Module):
    def __init__(self, embedding_dim:int=768,  mlp_size:int=3072, dropout:float=0.1): 
        super().__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim), 
            nn.Dropout(p=dropout) 
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768, 
                 num_heads:int=12, 
                 mlp_size:int=3072,
                 mlp_dropout:float=0.1, 
                 attn_dropout:float=0.0): 
        super().__init__()

        self.msa_block = MSABlock(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.mlp_block =  MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)

    def forward(self, x):
        x =  self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x

class ViT(nn.Module):
    def __init__(self,
                 attn_dropout:float=0.0, 
                 img_size:int=224, 
                 in_channels:int=3, 
                 patch_size:int=16, 
                 num_transformer_layers:int=12, 
                 embedding_dim:int=768, 
                 mlp_size:int=3072, 
                 num_heads:int=12, 
                 mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1, 
                 num_classes:int=1000): 
        super().__init__()

        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."

        self.num_patches = (img_size * img_size) // patch_size**2

        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim), requires_grad=True)

        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches+1, embedding_dim), requires_grad=True)

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim)

        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                            num_heads=num_heads,
                                                                            mlp_size=mlp_size,
                                                                            mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1) 

        x = self.patch_embedding(x)

        x = torch.cat((class_token, x), dim=1)

        x = self.position_embedding + x

        x = self.embedding_dropout(x)

        x = self.transformer_encoder(x)

        x = self.classifier(x[:, 0]) 

        return x

# if __name__ == '__main__':
    # import argparse
    # import yaml

    # parser = argparse.ArgumentParser(description='Process model size.')
    # parser.add_argument('--model_size', choices=['base', 'large', 'huge'], default='base',
    #                     help='Model size configuration (default: base)')
    # args = parser.parse_args()

    # with open('configs.yaml', 'r') as file:
    #     config = yaml.safe_load(file)

    # model_config = config.get(args.model_size)

    # if model_config is None:
    #     model_config = config['base']

    # print(f"Selected model size: {args.model_size}")
    # print(f"Layers: {model_config['layers']}")
    # print(f"Hidden size: {model_config['hidden_size']}")
    # print(f"MLP size: {model_config['mlp_size']}")
    # print(f"Heads: {model_config['heads']}")

    # vit = ViT(num_transformer_layers=model_config['layers'],
    #           embedding_dim=model_config['hidden_size'],
    #           mlp_size=model_config['mlp_size'],
    #           num_heads=model_config['heads'])
    
    # from torchinfo import summary
    # summary(model=vit,
    #     input_size=(32, 3, 224, 224),
    #     col_names=["input_size", "output_size", "num_params", "trainable"],
    #     col_width=20,
    #     row_settings=["var_names"]
    # )   

