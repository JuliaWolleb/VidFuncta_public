import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import math
import lpips
from common.utils import MetricLogger, psnr
from train.maml_boot import inner_adapt_test_scale_v2, inner_adapt_test_scale, inner_loop_step_tt_gradscale_v2, inner_adapt_test_scale_v3
import cv2

import torch
from torch.autograd import Variable
from compute_ssim import ssim3D, ssim
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_model(args, model_wrapper, test_loader, logger=None):
    metric_logger = MetricLogger(delimiter="  ")

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    model_wrapper.model.eval()
    model_wrapper.coord_init()

    lpips_score = lpips.LPIPS(net='alex').to(device)

    for n, data in enumerate(test_loader):
        data, _ = data
        if args.dimension == 'swap':
            data= torch.permute(data, (1,0,2, 3))   #load time points as batchsize
        if args.dimension == '3d':
            data = data[None,...].float().to(device, non_blocking=True) 
        print('data', data.shape)
        data = data.to(device)
        batch_size = data.size(0)
        if  data.isnan().any():
                        print('got nan')
                        continue

      


        model_wrapper.model.reset_modulations()
        if args.v_dim>0:
             model_wrapper.model.reset_vdim()

        _ = inner_adapt_test_scale(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                                   num_steps=args.inner_steps_test, first_order=True,
                                   sample_type=args.sample_type, scale_type='grad')  #used to be v3 for subsequent
        
      # model_wrapper.model.reset_vdim()
        plot = True
        with torch.no_grad():
            pred = model_wrapper().clamp(0, 1)
            print('pred', pred.shape)
            if  pred.isnan().any():
                        print('got nan')
                        continue
            if n < 10 and plot == True:
                    if args.dimension == '3d':
                        print('got 3d image')
                        to_pil = transforms.ToPILImage()
                        image = to_pil(data[0,0,2,...].squeeze())
                        image.save(f"./imgs_3d/{n}_input0.png")
                        image = to_pil(pred[0,0,2,...].squeeze())
                        image.save(f"./imgs_3d/{n}_recon0.png")

                        image = to_pil(data[0,0,5,...].squeeze())
                        image.save(f"./imgs_3d/{n}_input1.png")
                        image = to_pil(pred[0,0,5,...].squeeze())
                        image.save(f"./imgs_3d/{n}_recon1.png")

                        image = to_pil(data[0,0,9,...].squeeze())
                        image.save(f"./imgs_3d/{n}_input2.png")
                        image = to_pil(pred[0,0,9,...].squeeze())
                        image.save(f"./imgs_3d/{n}_recon2.png")




                    else:
                        to_pil = transforms.ToPILImage()
                        image = to_pil(data[0,...].squeeze())
                        image.save(f"./imgs/{n}_input0.png")
                        image = to_pil(pred[0,...].squeeze())
                        image.save(f"./imgs/{n}_recon0.png")

                        image = to_pil(data[1,...].squeeze())
                        image.save(f"./imgs/{n}_input1.png")
                        image = to_pil(pred[1,...].squeeze())
                        image.save(f"./imgs/{n}_recon1.png")


        if args.data_type == 'img':
           
            lpips_results = lpips_score((pred[:,:1,...] * 2 - 1), (data[:,:1,...] * 2 - 1)).mean()
            mse_results = F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean()
            psnr_results = psnr(
                F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean(dim=1)
            ).mean()
            ssim_results = ssim(pred, data, data_range=1.).mean()

        elif args.data_type == 'img3d':
            print('data', data.shape, 'pred', pred.shape)
            mse_results = F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean()
            psnr_results = psnr(
                F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean(dim=1)
            ).mean()
            ssim_results = ssim3D(pred, data).mean()
            lpips_results = torch.zeros_like(psnr_results)

        elif args.data_type == 'timeseries':
            mse_results = F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean()
            psnr_results = psnr(
                F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean(dim=1)
            ).mean()
            ssim_results = torch.zeros_like(psnr_results)
            lpips_results = torch.zeros_like(psnr_results)

        else:
            raise NotImplementedError()

        metric_logger.meters['lpips'].update(lpips_results.item(), n=batch_size)
        metric_logger.meters['psnr'].update(psnr_results.item(), n=batch_size)
        metric_logger.meters['mse'].update(mse_results.item(), n=batch_size)
        metric_logger.meters['ssim'].update(ssim_results.item(), n=batch_size)

        if n % 10 == 0:
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()

            log_(f'*[EVAL {n}][PSNR %.6f][LPIPS %.6f][SSIM %.6f][MSE %.6f]' %
                 (metric_logger.psnr.global_avg, metric_logger.lpips.global_avg,
                  metric_logger.ssim.global_avg, metric_logger.mse.global_avg))




    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log_(f'*[EVAL Final][PSNR %.8f][LPIPS %.8f][SSIM %.8f][MSE %.8f]' %
          (metric_logger.psnr.global_avg, metric_logger.lpips.global_avg,
           metric_logger.ssim.global_avg, metric_logger.mse.global_avg))


    return



def test_model_autoregressive(args, model_wrapper, test_loader, logger=None):
    metric_logger = MetricLogger(delimiter="  ")

    

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    model_wrapper.model.eval()
    model_wrapper.coord_init()

    lpips_score = lpips.LPIPS(net='alex').to(device)

    for n, data in enumerate(test_loader):
        
        model_wrapper.model.reset_modulations()
        if args.v_dim > 0:
            model_wrapper.model.reset_vdim()
        data, label, fps, name = data
        print('name', name[0])
       # if '0X1AA8E4C0155A02E0' not in name[0]:
      #  if '0X1AA8E4C0155A02E0' in name[0]:
        #    print('found it')
       # if '0X1AA8E4C0155A02E0' not in name[0]:
        #    continue
        print('data', data.shape, 'label', label)
        if args.dimension == 'swap':
            data = torch.permute(data, (1,0,2, 3))   #load time points as batchsize
        print('data', data.shape)  #should be of length 60
        num_steps =math.floor(data.shape[0]/args.num_frames)  #should be 15
        print('num_steps', num_steps)
        data = data.float().to(device)
        batch_size = data.size(0)
        if  data.isnan().any():
                        print('got nan')
                        continue
        pred =torch.zeros(data.shape)
        print('clip', pred.shape)
       # target = torch.zeros(60, args.input_dim)
        for k in range(int(num_steps)):
           
            if k ==0:
                model_wrapper.model.reset_modulations()
                model_wrapper.model.reset_vdim()
                chunk = data[0:args.num_frames,...]
                print('chunk', chunk.shape)
              
                if args.mode == 'concat' or args.mode =='separate':
                    print('adapt scale v2')
                    _ = inner_adapt_test_scale_v2(model_wrapper=model_wrapper, data=chunk, step_size=args.inner_lr,
                                        num_steps=args.inner_steps_test, first_order=True,
                                        sample_type=args.sample_type, scale_type='grad')
                if args.mode == 'additive':
                    _ = inner_adapt_test_scale_v3(model_wrapper=model_wrapper, data=chunk, step_size=args.inner_lr,
                                        num_steps=args.inner_steps_test, first_order=True,
                                        sample_type=args.sample_type, scale_type='grad')
                if args.mode == 'spatial':
                    _ = inner_adapt_test_scale(model_wrapper=model_wrapper, data=chunk, step_size=args.inner_lr,
                                        num_steps=args.inner_steps_test, first_order=True,
                                        sample_type=args.sample_type, scale_type='grad')

              
                with torch.no_grad():
                    inter = model_wrapper().clamp(0, 1)
                    pred[0:args.num_frames,...] = inter

            else:
                model_wrapper.model.reset_modulations()
             #   model_wrapper.model.reset_vdim()

                chunk = data[k*args.num_frames: (k+1)*args.num_frames ,...]
                print('adapt test scale')
                
                _ = inner_adapt_test_scale(model_wrapper=model_wrapper, data=chunk, step_size=args.inner_lr,
                                        num_steps=args.inner_steps_test, first_order=True,
                                        sample_type=args.sample_type, scale_type='grad')
                
                with torch.no_grad():
                 #   model_wrapper.model.reset_vdim()
                    inter = model_wrapper().clamp(0, 1)
                    pred[k*args.num_frames: (k+1)*args.num_frames ,...] = inter
        
      #  to_pil = transforms.ToPILImage()
     #   model_wrapper.model.reset_modulations()
      #  predv = model_wrapper().clamp(0, 1)
      #  print('predv', predv.shape)
      #  print('data', data.shape, 'pred', pred.shape)
      #  image = to_pil(predv[0])
      #  image.save(f"./imgs_add/predicted_v.png")
      #  sys.exit('rer')
        
        # mean_frame_org = torch.mean(data, dim=0)
        # print('meanframe', mean_frame_org.shape)
        # image = to_pil(mean_frame_org[0])
        # image.save(f"./imgs_add/mean_frame_org.png")
        # mean_frame_pred = torch.mean(pred, dim=0)
        # image = to_pil(mean_frame_pred[0])
        # image.save(f"./imgs_add/mean_frame_pred.png")
        
        data = data[:num_steps*args.num_frames, ...]
        pred = pred[:num_steps*args.num_frames, ...]

        print('data3d', data.shape, pred.shape)

        batch_size = data.shape[0]
        plot = False
        with torch.no_grad():
           
            if  pred.isnan().any():
                        print('got nan')
                        continue
            if plot == True:
                for k in range(1):
                    print('data', data.shape)
                    to_pil = transforms.ToPILImage()
                    image = to_pil(data[1*k,...].squeeze())
                    image.save(f"./imgs/{n}_input.png")
                    image = to_pil(pred[1*k,...].squeeze())
                    image.save(f"./imgs/{n}_recon.png")

        pred = pred.to(device)
        data = data.to(device)
        
        if args.data_type == 'img':
            print('data', data.shape, pred.shape)
            lpips_results = lpips_score((pred[:,:1,...] * 2 - 1), (data[:,:1,...] * 2 - 1)).mean()
            mse_results = F.mse_loss(data.view(1, -1), pred.reshape(1, -1), reduce=False).mean()
            psnr_results = psnr(
                F.mse_loss(data.view(1, -1), pred.reshape(1, -1), reduce=False).mean(dim=1)
            ).mean()
            ssim_results = ssim(pred, data).mean()
          


        elif args.data_type == 'img3d':
            mse_results = F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean()
            psnr_results = psnr(
                F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean(dim=1)
            ).mean()
            ssim_results = ssim3D(pred, data).mean()
            lpips_results = torch.zeros_like(psnr_results)

        elif args.data_type == 'timeseries':
            mse_results = F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean()
            psnr_results = psnr(
                F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean(dim=1)
            ).mean()
            ssim_results = torch.zeros_like(psnr_results)
            lpips_results = torch.zeros_like(psnr_results)

        else:
            raise NotImplementedError()

        metric_logger.meters['lpips'].update(lpips_results.item(), n=batch_size)
        metric_logger.meters['psnr'].update(psnr_results.item(), n=batch_size)
        metric_logger.meters['mse'].update(mse_results.item(), n=batch_size)
        metric_logger.meters['ssim'].update(ssim_results.item(), n=batch_size)

        if n % 1 == 0:
            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            print('label', label)

            log_(f'*[EVAL {n}][PSNR %.6f][LPIPS %.6f][SSIM %.6f][MSE %.6f]' %
                 (metric_logger.psnr.global_avg, metric_logger.lpips.global_avg,
                  metric_logger.ssim.global_avg, metric_logger.mse.global_avg))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log_(f'*[EVAL Final][PSNR %.8f][LPIPS %.8f][SSIM %.8f][MSE %.8f]' %
         (metric_logger.psnr.global_avg, metric_logger.lpips.global_avg,
          metric_logger.ssim.global_avg, metric_logger.mse.global_avg))

    return




def reconstruct_model_autoregressive(args, model_wrapper, test_loader, logger=None,  set=None):
    metric_logger = MetricLogger(delimiter="  ")
    print('set', set)
    

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    model_wrapper.model.eval()
    model_wrapper.coord_init()

   # lpips_score = lpips.LPIPS(net='alex').to(device)

    for n, data in enumerate(test_loader):
     # if n > 3475:
        model_wrapper.model.reset_modulations()
        if args.v_dim > 0:
            model_wrapper.model.reset_vdim()
        data, label, fps, name = data
        print('name', name)
        fps=int(fps)
        print('data', data.shape, 'label', label)

        if args.dimension == 'swap':
            print('data swap', data.shape)
            data = torch.permute(data, (1,0,2, 3))   #load time points as batchsize
       
       
        data = data.float().to(device)
        batch_size = data.size(1)
        time = data.size(0)

        if args.dimension == '3d':
            batch_size = data.size(0)
            time = data.size(1)
        print('time', time, 'batch', batch_size)
        num_steps =math.floor(time/args.num_frames)  
        print('num_steps', num_steps)
        if  data.isnan().any():
                        print('got nan')
                        continue
        pred =torch.zeros(data.shape)
        print('pred', pred.shape)
        print('clip', pred.shape, 'time', time)
        target = torch.zeros(time, args.latent_modulation_dim)
        if args.mode == 'spatial':
          target = torch.zeros(time, 64,4,4)
        for k in range(int(num_steps)):
           
            if k ==0:
                model_wrapper.model.reset_modulations()
                if args.v_dim >0:
                    model_wrapper.model.reset_vdim()
                chunk = data[0:args.num_frames,...]
                if args.dimension == '3d':
                    chunk = data[:,0:args.num_frames,...][None,...]
                

                elif  args.mode =='separate':
                    _ = inner_adapt_test_scale_v2(model_wrapper=model_wrapper, data=chunk, step_size=args.inner_lr,
                                        num_steps=args.inner_steps_test, first_order=True,
                                        sample_type=args.sample_type, scale_type='grad')
                    v = model_wrapper.model.vdim.detach().cpu()
                    print('got v', v.shape)

             
                elif args.mode == 'plain' or args.mode =='spatial':
                    print('got plain')
                    _ = inner_adapt_test_scale(model_wrapper=model_wrapper, data=chunk, step_size=args.inner_lr,
                                        num_steps=args.inner_steps_test, first_order=True,
                                        sample_type=args.sample_type, scale_type='grad')
                target[0:args.num_frames,...] = model_wrapper.model.modulations.detach().cpu()
              
                with torch.no_grad():
                    inter = model_wrapper(t=label).clamp(0, 1)
                    
                    if args.dimension == '3d':
                        pred[:,0:args.num_frames,...] = inter
                    else:
                        pred[0:args.num_frames,...] = inter

            else:
                model_wrapper.model.reset_modulations()

                chunk = data[k*args.num_frames: (k+1)*args.num_frames ,...]
                if args.dimension == '3d':
                    chunk = data[:,k*args.num_frames: (k+1)*args.num_frames,...][None,...]
                
      
                _ = inner_adapt_test_scale(model_wrapper=model_wrapper, data=chunk, step_size=args.inner_lr,
                                        num_steps=args.inner_steps_test, first_order=True,
                                        sample_type=args.sample_type, scale_type='grad')
                target[k*args.num_frames: (k+1)*args.num_frames ,...] = model_wrapper.model.modulations.detach().cpu()

                with torch.no_grad():
                    inter = model_wrapper(x=None, t=label).clamp(0, 1)
                    if args.dimension == '3d':
                        pred[:,k*args.num_frames: (k+1)*args.num_frames,...] = inter
                    else:
                        
                        pred[k*args.num_frames: (k+1)*args.num_frames ,...] = inter
        
        data = data[:num_steps*args.num_frames, ...]
        pred = pred[:num_steps*args.num_frames, ...]
    

        plot = True
        pred = pred.to(device)
        data = data.to(device)

        
        if args.data_type == 'img':
            mse_results = F.mse_loss(data.view(1, -1), pred.reshape(1, -1), reduce=False).mean()
            psnr_results = psnr(
                F.mse_loss(data.view(1, -1), pred.reshape(1, -1), reduce=False).mean(dim=1)
            ).mean()
            ssim_results = ssim(pred, data, ).mean()
            ssim_3d = ssim3D(pred.permute(1,0,2,3)[None,...], data.permute(1,0,2,3)[None,...])
            print('psnr', psnr_results, 'ssim3d', ssim_3d)

        elif args.data_type == 'img3d':
            print('data', data.shape, 'pred', pred.shape)
            mse_results = F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean()
            psnr_results = psnr(
                F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean(dim=1)
            ).mean()
            print('pred', pred.shape, 'data', data.shape)
            ssim_3d = ssim3D(pred[None,...], data[None,...]).mean()

        elif args.data_type == 'timeseries':
            mse_results = F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean()
            psnr_results = psnr(
                F.mse_loss(data.view(batch_size, -1), pred.reshape(batch_size, -1), reduce=False).mean(dim=1)
            ).mean()
            ssim_results = torch.zeros_like(psnr_results)

        else:
            raise NotImplementedError()

        metric_logger.meters['psnr'].update(psnr_results.item(), n=batch_size)
        metric_logger.meters['mse'].update(mse_results.item(), n=batch_size)
        metric_logger.meters['ssim3d'].update(ssim_3d.item(), n=batch_size)
        if args.v_dim >0:
            v= v.detach().cpu()
        else: 
            v=0

        datapoint = {
                'modulations': target.detach().cpu(),
                'label': label,
                'v': v,
                'name': name,
                'psnr': psnr_results.item(),
                'ssim3d': ssim_3d.item(),
            }
        print('name', name)
        filename = name[0].split('/')[-1].split('.')[0]
        print('filename', filename)
        sdir = args.save_dir + 'nfset'+ f'/{set}/' + filename + f'_datapoint_{(n * batch_size)}.pt'
        print('sdir', sdir)
        torch.save(datapoint, sdir)
        print('saved datapoint' , sdir, n)

      
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('label', label)

        log_(f'*[EVAL {n}][PSNR %.6f][SSIM3D %.6f][MSE %.6f]' %
                 (metric_logger.psnr.global_avg, 
                   metric_logger.ssim3d.global_avg, metric_logger.mse.global_avg))

        save_videos = True
        #save the videos
        if save_videos:
            tensor = pred.cpu()
            print('tensor', tensor.shape)
            outputdir = args.save_dir +'/videos'
            outputname = name[0].split('/')[-1]
            print('outputname', outputname)
            outputname = outputname.split('.')[0]+'.avi'
            print('outputname2', outputname)
            savepath= os.path.join(outputdir,outputname)
            print('savepath', savepath)
            tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # Normalize to [0,1]
            tensor = (tensor * 255).byte()  # Convert to [0,255] uint8
            if args.dimension == '3d':
                tensor = tensor.squeeze(0)
            # Convert to numpy and reshape to (H, W) grayscale frames
            frames = tensor.squeeze(1).numpy()  # Shape: (60, 112, 112)


            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
            frame_size = (112, 112)
            out = cv2.VideoWriter(savepath, fourcc, fps, frame_size, isColor=True)

            if not out.isOpened():
                raise RuntimeError("Failed to open video writer")

            # Write frames
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                out.write(frame_bgr)

            out.release()
            print("Saved as output.avi")

        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log_(f'*[EVAL Final][PSNR %.8f][SSIM3d %.8f][MSE %.8f]' %
         (metric_logger.psnr.global_avg, 
         metric_logger.ssim3d.global_avg,  metric_logger.mse.global_avg))

    return