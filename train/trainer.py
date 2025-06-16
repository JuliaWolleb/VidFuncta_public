import argparse
import time
import torch
from common.utils import resume_training, MetricLogger, save_checkpoint, save_checkpoint_step

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trainer(args, train_function, test_function, model_wrapper, meta_optimizer, train_loader, test_loader, logger):
    """
    The main function that performs the training. Iteratively calls training steps (train_function) and evaluations (test_function).
    """
    metric_logger = MetricLogger(delimiter="  ")

    """ Resume training (optional with '--resume_path' flag) """
    is_best, start_step, best_psnr, psnr = resume_training(args, model_wrapper, meta_optimizer)

    """ Start Training """
    logger.log_dirname(f"Start training")

    for it, train_batch in enumerate(train_loader):
        step = start_step + it + 1
        if step > args.outer_steps:
            break
       
        try:
            train_batch, label_batch,_,_ = train_batch 
        except:
            continue
       # if label_batch==2:
         #   continue
        if args.dimension == 'sobel':
            train_batch = train_batch[0,...]
           
        elif args.dimension == 'swap':
            if args.mode == 'time': 
                train_batch['vid'] = torch.permute(train_batch['vid'], (1, 0, 2, 3))
                train_batch['vid'] = train_batch['vid'].float().to(device, non_blocking=True) 
                if torch.isnan(train_batch['vid']).any():
                    continue
        
                train_batch['time'] = train_batch['time'].float().to(device, non_blocking=True) 
            else:
                if torch.isnan(train_batch).any():
                    continue
                train_batch = torch.permute(train_batch, (1, 0, 2, 3))

                train_batch = train_batch.float().to(device, non_blocking=True)  # Batch of images bs, 1, img_size, img_size
        if args.dimension == '3d':
            if torch.isnan(train_batch).any():
                    continue
            
            try:
                train_batch = train_batch[None,...].float().to(device, non_blocking=True) 
                if train_batch.shape[2] <10:
                    tiled = train_batch.repeat(1, 1, 3, 1, 1)  # repeat along time axis
                    train_batch = tiled[:, :, :10, :, :] 
                    print('trainbatch', train_batch.shape)
            except:
                continue
        if args.guidance == True or args.comment == 'separate2':
            if torch.isnan(train_batch).any():
                continue
            train_batch= {
                    "vid": train_batch,
                # "sites": sites,
                #  "mask": mask,
                 #   'time': timestep,
                    'label': label_batch.float().to(device, non_blocking=True) 
                }
        


        """ Perform training step """
       
        train_function(args, step, model_wrapper, meta_optimizer, train_batch, metric_logger, logger)
     

        """ Evaluate and save model """
        if step % args.eval_step == 0:
            psnr = test_function(args, step, model_wrapper, test_loader, logger)

            if best_psnr < psnr:
                best_psnr = psnr
                save_checkpoint(args, step, best_psnr, model_wrapper, meta_optimizer.state_dict(), logger.logdir,
                                is_best=True)

            logger.scalar_summary('eval/best_psnr', best_psnr, step)
            logger.log('[EVAL] [Step %3d] [PSNR %5.2f] [BestPSNR %5.2f]' % (step, psnr, best_psnr))

        """ Save model every save_step steps"""
        if step % args.save_step == 0:
            save_checkpoint_step(args, step, best_psnr, model_wrapper, meta_optimizer.state_dict(), logger.logdir)

    """ Save the last model"""
    save_checkpoint(args, args.outer_steps, best_psnr, model_wrapper, meta_optimizer.state_dict(), logger.logdir)