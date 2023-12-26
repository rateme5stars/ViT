def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

if __name__ == '__main__':
    from timeit import default_timer as timer
    import argparse
    import yaml
    from model.ViT import ViT
    from utils.dataloader import create_dataloader
    from utils.engine import train
    import torch

    training_config = load_config('config.yaml')
    model_config = load_config('model/models.yaml')

    parser = argparse.ArgumentParser(description="Configurations")
    parser.add_argument('--epochs', default=training_config['training']['epoch'], help='Number of epoch')
    parser.add_argument('--lr', default=training_config['training']['learning_rate'], help='learning_rate')
    parser.add_argument('--batch_size', default=training_config['training']['batch_size'], help='batch_size')
    parser.add_argument('--weight_decay', default=training_config['training']['weight_decay'], help='weight_decay')
    parser.add_argument('--betas', default=training_config['training']['betas'], help='betas in form "b1 b2"')

    parser.add_argument('--model_size', choices=['base', 'large', 'huge'], default='base',
                        help='Model size configuration (default: base)')
    args = parser.parse_args()
    
    archi_config = model_config.get(args.model_size)
    if model_config is None:
        model_config = archi_config['base']

    print(f"Selected ViT model size: {args.model_size}")
    print(f"Layers: {archi_config['layers']}")
    print(f"Hidden size: {archi_config['hidden_size']}")
    print(f"MLP size: {archi_config['mlp_size']}")
    print(f"Heads: {archi_config['heads']}")  

    train_loader, test_loader, classes = create_dataloader(args.batch_size)

    vit = ViT(num_transformer_layers=archi_config['layers'],
              embedding_dim=archi_config['hidden_size'],
              mlp_size=archi_config['mlp_size'],
              num_heads=archi_config['heads'], 
              num_classes=len(classes))

    betas = tuple(map(float, args.betas.split(' ')))
    optimizer = torch.optim.Adam(params=vit.parameters(), lr=args.lr, betas=betas, weight_decay=args.weight_decay)  

    loss_fn = torch.nn.CrossEntropyLoss()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    start = timer()
    train(vit, train_loader, test_loader, optimizer, loss_fn, device, args.epochs)
    end = timer()
    print(f'Training time: {end-start}s on {device}')

    