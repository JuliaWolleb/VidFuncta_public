import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from common.utils import set_random_seed, load_model
from data import dataset_echonet
from common.args import parse_args
from common.utils import set_random_seed, Logger, InfiniteSampler
from data.dataset import get_dataset
from models.inrs_spatial import LatentModulatedSIREN_spatial_plain, LatentModulatedSIREN_v2, LatentModulatedSIREN_v3, LatentModulatedSIREN_v4
from models.wire import INR
from models.model_wrapper import ModelWrapper
from train.trainer import trainer
from train.maml_boot import train_step
from eval.maml_scale import test_model


def main(args):
    """
    Main function to call for running a training procedure.
    :param args: parameters parsed from the command line.
    :return: Nothing.
    """

    """ Set a device to use """
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    args.data_size = (1,args.img_size, args.img_size)
    if args.dimension == '3d':
        args.data_size = (1, args.num_frames, args.img_size, args.img_size)


    """ Enable determinism """
   # set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print('dataset', args.dataset)
    """ Define dataset """
    if args.dataset == 'framewise_echo' or args.dataset == 'echo_3d':
        print('got 3d dataset')
        infinite_sampler = InfiniteSampler(dataset_echonet.Echo(split="train", period=8, length=args.num_frames  ), rank=0, num_replicas=1, shuffle=True, seed=args.seed)
        train_loader = torch.utils.data.DataLoader(dataset_echonet.Echo(split="train", period=8, length=args.num_frames  ), batch_size=args.batch_size, sampler = infinite_sampler, num_workers=4, prefetch_factor=2)
        val_loader = torch.utils.data.DataLoader(dataset_echonet.Echo(split="val",  period=8, length=args.num_frames  ), batch_size=args.test_batch_size)
        test_loader = torch.utils.data.DataLoader(dataset_echonet.Echo(split="test", period=8, length=args.num_frames  ), batch_size=args.test_batch_size)
        if args.dimension == '3d':
            args.data_size = (1, args.num_frames, args.img_size, args.img_size)

    else:
        
        """ Define dataloader """
        train_set, val_set, test_set = get_dataset(args, all = True)

        
        if args.finetune == True:
            infinite_sampler = InfiniteSampler(val_set, rank=0, num_replicas=1, shuffle=True)
            train_loader = DataLoader(val_set, sampler=infinite_sampler, batch_size=args.batch_size, num_workers=4,
                                prefetch_factor=2, drop_last=True)

            print('length to finetune', len(val_set))
        else: 
            train_set,  test_set = get_dataset(args)

            infinite_sampler = InfiniteSampler(train_set, rank=0, num_replicas=1, shuffle=True)

            train_loader = DataLoader(train_set, sampler=infinite_sampler, batch_size=args.batch_size, num_workers=4,
                                prefetch_factor=2, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=4 , drop_last=True)
       
    """ Initialize model and optimizer """
    

    model = LatentModulatedSIREN_spatial_plain(
            in_size=args.in_size,
            out_size=args.out_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            latent_modulation_dim=args.latent_modulation_dim,
            w0=args.w0,
            w0_increments=args.w0_increment,
            modulate_shift=args.modulate_shift,
            modulate_scale=args.modulate_scale,
            enable_skip_connections=args.enable_skip_connections,
            guidance = args.guidance
        ).to(device)
    print('got spatial siren')
    model.modulations = torch.zeros(size=[args.num_frames, 64, 4 , 4], requires_grad=True).to(device)
   # model.vdim = torch.zeros(size=[1, 64, 4, 4], requires_grad=True).to(device)

    print('modulations init', model.modulations.shape)
      


    model = ModelWrapper(args, model)
    load_model(args, model)
    print('loaded model')

    meta_optimizer = optim.AdamW(params=model.parameters(), lr=args.meta_lr)

    """ Define training and test functions """
    train_function = train_step
    test_function = test_model

    """ Define logger """
    fname = (f'{args.dataset}_model{args.model}_numlayers{args.num_layers}_option{args.option}_guidance{args.guidance}_lam{args.lam}_inner{args.inner_steps}_size{args.img_size}_innerlr{args.inner_lr}_meatlr{args.meta_lr}_latentsize{args.shape}_gamma{args.data_ratio}_hiddensize{args.hidden_size}_vdim{args.v_dim}_frames{args.num_frames}_numlayers{args.num_layers}_comment{args.comment}'
             f'{args.config.split("/")[-1].split(".yaml")[0]}')
    logger = Logger(fname, ask=args.resume_path is None, rank=args.gpu_id)
    logger.log(args)
    logger.log(model)

    """ Perform training """
    trainer(args, train_function, test_function, model, meta_optimizer, train_loader, test_loader, logger)

    """ Close logger """
    logger.close_writer()


if __name__ == "__main__":
    args = parse_args()
    print('args', args)
    main(args)
