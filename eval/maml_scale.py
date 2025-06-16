import torch
import matplotlib.pyplot as plt
from common.utils import MetricLogger, psnr
from train.maml_boot import inner_adapt_test_scale, inner_adapt_test_scale_v3, inner_adapt_test_scale_v2, inner_adapt_test_scale_time, inner_adapt_test_scale_vae


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(args, step, model_wrapper, test_loader, logger=None):
    metric_logger = MetricLogger(delimiter="  ")

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    model_wrapper.model.eval()
    model_wrapper.coord_init()

    for n, data in enumerate(test_loader):
        if n * args.num_frames > args.num_test_signals:
            break
        print('n', n)
        data, label,_,_ = data
        print('label', label)
      #  if label == 2:
          #  print('skip breast')
         #   continue
         

        if args.dimension == 'swap':
            if args.mode == 'time': 
                data['vid'] = torch.permute(data['vid'], (1, 0, 2, 3))
                data['vid'] = data['vid'].float().to(device, non_blocking=True) 
                if data['vid'].isnan().any():
                    print('got nan')
                    continue
                data['time'] = data['time'].float().to(device, non_blocking=True) 
                batch_size = data['vid'].size(0)
            else:
                if data.isnan().any():
                    print('got nan')
                    continue
              #  if label.isnan().any():
                 #   print('got nan')
                #   continue
                data = data.float().to(device, non_blocking=True) 
                data = torch.permute(data, (1, 0, 2, 3))
                batch_size = data.size(0)
        elif args.dimension == 'sobel':
             data = data[0,...]

             data = data.float().to(device)
        elif args.dimension == '3d':
            data = data[None,...].float().to(device, non_blocking=True) 
            batch_size = data.size(0)
        
        model_wrapper.model.reset_modulations()
        if args.v_dim >0:
            model_wrapper.model.reset_vdim()


        if n == 3:
            if args.mode == 'time':
                input = data['vid']
            else:
                input = data
                print('input', input.shape)
        input = data
        if args.mode == 'time':
            loss_in_tt_gradscale = inner_adapt_test_scale_time(model_wrapper=model_wrapper, Data=data, step_size=args.inner_lr,
                                                      num_steps=args.inner_steps_test, first_order=True,
                                                      sample_type=args.sample_type, scale_type='grad')

        elif args.mode == 'vae':
            loss_in_tt_gradscale = inner_adapt_test_scale_vae(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                                                      num_steps=args.inner_steps_test, first_order=True,
                                                      sample_type=args.sample_type, scale_type='grad')


        elif args.comment == 'separate2':
            train_batch= {
                    'vid': data,
                    'label': label.float().to(device, non_blocking=True) 
                }
            loss_in_tt_gradscale = inner_adapt_test_scale_v2(model_wrapper=model_wrapper, data=train_batch, step_size=args.inner_lr,
                                                      num_steps=args.inner_steps_test, first_order=True,
                                                      sample_type=args.sample_type, scale_type='grad')
       
        elif args.v_dim > 0:

            print('gradscale v2')
            loss_in_tt_gradscale = inner_adapt_test_scale_v2(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                                                      num_steps=args.inner_steps_test, first_order=True,
                                                      sample_type=args.sample_type, scale_type='grad')

        else: 
            loss_in_tt_gradscale = inner_adapt_test_scale(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                                                      num_steps=args.inner_steps_test, first_order=True,
                                                      sample_type=args.sample_type, scale_type='grad')
        psnr_in_tt_gradscale = psnr(loss_in_tt_gradscale)
   

        """ Outer loss aggregation """
        with torch.no_grad():
            if args.mode == 'time':
                loss_out_tt_gradscale = model_wrapper(data['vid'], data['time'])
            elif args.comment == 'separate2':
                loss_out_tt_gradscale = model_wrapper(train_batch)
                print('got outer loss')
            
            else:
                loss_out_tt_gradscale = model_wrapper(data)
            psnr_out_tt_gradscale = psnr(loss_out_tt_gradscale)
           
            if n == 0:
                out = model_wrapper()

        metric_logger.meters['loss_inner_tt_gradscale'].update(loss_in_tt_gradscale.mean().item(), n=batch_size)
        metric_logger.meters['loss_outer_tt_gradscale'].update(loss_out_tt_gradscale.mean().item(), n=batch_size)
        metric_logger.meters['psnr_inner_tt_gradscale'].update(psnr_in_tt_gradscale.mean().item(), n=batch_size)
        metric_logger.meters['psnr_outer_tt_gradscale'].update(psnr_out_tt_gradscale.mean().item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    log_('*[EVAL Gradscale TestTime][LossInnerGSTT %.3f][LossOuterGSTT %.3f][PSNRInnerGSTT %.3f][PSNROuterGSTT %.3f]' %
         (metric_logger.loss_inner_tt_gradscale.global_avg, metric_logger.loss_outer_tt_gradscale.global_avg,
          metric_logger.psnr_inner_tt_gradscale.global_avg, metric_logger.psnr_outer_tt_gradscale.global_avg))

    logger.scalar_summary('eval/loss_inner_TT_gradscale', metric_logger.loss_inner_tt_gradscale.global_avg, step)
    logger.scalar_summary('eval/loss_outer_TT_gradscale', metric_logger.loss_outer_tt_gradscale.global_avg, step)
    logger.scalar_summary('eval/psnr_inner_TT_gradscale', metric_logger.psnr_inner_tt_gradscale.global_avg, step)
    logger.scalar_summary('eval/psnr_outer_TT_gradscale', metric_logger.psnr_outer_tt_gradscale.global_avg, step)
    out = model_wrapper()

    print('out', out.shape)
    logger.log_image('eval/img_in', input[:,:1,...], step)
    logger.log_image('eval/img_adapt_tt', out[:,:1,...], step)
    if args.sobel == True:
        logger.log_image('eval/img_sobel_in', input[:,1:,...], step)
        logger.log_image('eval/img_adapt_tt_sobel', out[:,1:,...], step)
    input = input.detach().cpu()
    # plt.figure()
    # plt.title('sobel')
    # plt.imshow(input[0,1,...], cmap='gray')  # Tensor format: [batch, channel, height, width]
    # plt.savefig("sobel.png") 
    # plt.figure()
    # plt.title('frame')
    # plt.imshow(input[0,0,...], cmap='gray')  # Tensor format: [batch, channel, height, width]
    # plt.savefig("frame.png") 


    return metric_logger.psnr_outer_tt_gradscale.global_avg