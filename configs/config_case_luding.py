cfg = dict(
    train = dict(
        total_epoch   = 150,
        cuda          = True,
        pretrain      = False,
        resume        = False,
        freeze_param  = False,
        ckpt_savepath = 'checkpoints/luding/scdunetpp',                                                      # path for resume
        ckpt_resume   = 'checkpoints/luding/scdunetpp/epc125-trloss0.298-valoss0.359-iou0.733-f1-0.846.pth', # checkpoint for resume
        ckpt_test     = 'epc121-trloss0.296-valoss0.330-iou0.722-f1-0.839.pth'                               # checkpoint for test set
    ),
    model = dict(
        model_type   = 'scdunetpp',
        num_classes  = 2 
    ),
    dataset = dict(
        dataset_name = 'luding',
        dataset_path = 'data/luding/DATA', 
        train_lines  = 'data/luding/DATA/config/train.txt',
        val_lines    = 'data/luding/DATA/config/val.txt',
        test_lines   = 'data/luding/DATA/config/test.txt'
    ),
    dataloader = dict(
        isShuffle    = True,
        batch_size   = 8,
        num_workers  = 2,
        input_shape  = (128,128),
        in_channels  = 21,       
        isOnLineAug  = True  
    ),
    optimizer = dict(
        base_lr      = 1e-4,    
        min_lr       = 9e-6,
        step_size    = 1,
        gamma        = 0.9,   
        weight_decay = 1e-4 
    )
)
