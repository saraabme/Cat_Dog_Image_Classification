
def train(model, log_dir=None, train_data=None, valid_data=None, optimizer=None, batch_size=128, resize=None, n_epochs=10,
          device=None, is_resnet=False, log_string=None, schedule_lr=False):
    import torch
    import torch.utils.tensorboard as tb
    from data import load
    import numpy as np


    # Load training and validation data if not provided
    if train_data is None:
        train_data = load.get_dogs_and_cats(resize=resize, batch_size=batch_size, is_resnet=is_resnet)
    if valid_data is None:
        valid_data = load.get_dogs_and_cats('valid', resize=resize, batch_size=batch_size, is_resnet=is_resnet)
    
    # Initialize optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    
    # Learning rate scheduler to reduce learning rate based on validation performance
    if schedule_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=50)

    # Transfer the data to a GPU (optional)
    if device is not None:
        model = model.to(device)

    # Initialize tensorboard loggers for training and validation metrics
    train_logger, valid_logger = None, None
    if log_dir is not None:
        train_logger = tb.SummaryWriter(log_dir+"/train", flush_secs=5)
        valid_logger = tb.SummaryWriter(log_dir+"/valid", flush_secs=5)
        if log_string is not None:
            train_logger.add_text("info", log_string)

    # Construct the loss and accuracy functions
    loss = torch.nn.BCEWithLogitsLoss()
    accuracy = lambda o, l: ((o > 0).long() == l.long()).float()

    # Train the network
    global_step = 0
    for epoch in range(n_epochs):

        accuracies = []
        for it, (data, label) in enumerate(train_data):
            # Transfer the data to a GPU (optional)
            if device is not None:
                data, label = data.to(device), label.to(device)

            # Forward pass: compute predicted outputs by passing inputs to the model
            o = model(data)

            # Calculate the batch loss and accuracy
            loss_val = loss(o, label.float())
            accuracies.extend(accuracy(o, label).detach().cpu().numpy())

            # Logging the training loss
            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step=global_step)

            # Take a gradient step
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            global_step += 1

        # Log training accuracy after each epoch
        if train_logger is not None:
            train_logger.add_scalar('accuracy', np.mean(accuracies), global_step=global_step)

        # Validation loop
        val_accuracies = []
        for it, (data, label) in enumerate(valid_data):
            # Transfer the data to a GPU (optional)
            if device is not None:
                data, label = data.to(device), label.to(device)
            # Produce the output - Forward pass to compute validation accuracy
            o = model(data)
            val_accuracies.extend(accuracy(o, label).detach().cpu().numpy())

        # Log and Update the LR
        if schedule_lr:
            train_logger.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=global_step)
            scheduler.step(np.mean(val_accuracies))

        # Log validation accuracy or print to console if no logger
        if valid_logger is not None:
            valid_logger.add_scalar('accuracy', np.mean(val_accuracies), global_step=global_step)
        else:
            print('epoch = % 3d   train accuracy = %0.3f   valid accuracy = %0.3f'%(epoch, np.mean(accuracies), np.mean(val_accuracies)))


if __name__ == "__main__":
    import torch
    # # Set up command-line arguments
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('logdir')
    parser.add_argument('-b', '--batch_size', type=int, default=128)
    parser.add_argument('--no_normalization', action='store_true')
    parser.add_argument('-n', '--n_epochs', type=int, default=10)
    parser.add_argument('-o', '--optimizer', default='optim.Adam(parameters)')
    parser.add_argument('-sl', '--schedule_lr', action='store_true')
    args = parser.parse_args()

    # Create the CUDA device if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create the ConvNet
    from model import ConvNet
    net = ConvNet()

    # Parse and evaluate the optimizer from string
    optimizer = eval(args.optimizer, {'parameters': net.parameters(), 'optim': torch.optim})

    # Load data with potential augmentations
    from data import load
    train_data = load.get_dogs_and_cats(batch_size=args.batch_size, random_crop=(128, 128), random_horizontal_flip=True,
                                        normalize=not args.no_normalization)
    valid_data = load.get_dogs_and_cats('valid', resize=(128, 128), batch_size=args.batch_size,
                                        normalize=not args.no_normalization)
    # Begin training
    train(net, args.logdir, train_data=train_data, valid_data=valid_data, device=device, resize=(128, 128),
          n_epochs=args.n_epochs, optimizer=optimizer, log_string=str(args), schedule_lr=args.schedule_lr)
