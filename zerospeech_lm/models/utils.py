import os
import torch
import torch.optim as optim


def save_checkpoint(model, optimizer, epoch, args):
    path = os.path.join(args.save_dir, 'checkpoints', str(epoch))
    if not os.path.exists(path):
        os.makedirs(path)
    if args.nGPU == 1:
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(path, 'model.pt'))
    else:
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(path, 'model.pt'))
    print('Model saved to {}!'.format(path))
    

def load_checkpoint(model, args):
    checkpoint = torch.load(os.path.join(args.load_dir, 'model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print('Model from {} loaded!\n'.format(args.load_dir))
    return model, optimizer, epoch


def reset_hidden(args, eval=False):
    if eval:
        bsz = 1
    else:
        bsz = args.bsz

    if args.arch == 'LSTM':
        if args.bidirectional:
            num_layers = args.num_layers * 2
            hidden_dim = args.hidden_dim // 2
        else:
            num_layers = args.num_layers
            hidden_dim = args.hidden_dim
        
        return (torch.zeros(bsz, num_layers, hidden_dim).to(args.device),
                torch.zeros(bsz, num_layers, hidden_dim).to(args.device)
                )

    if args.arch == 'QRNN':
        return torch.zeros(bsz, args.num_layers, args.hidden_dim).to(args.device)