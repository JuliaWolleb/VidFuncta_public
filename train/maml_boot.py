import torch
from common.utils import psnr
import sklearn
import torch.optim as optim
def modulation_consistency(modulations, modulations_bootstrapped, bs):
    """
    A function that calculates the L2-distance between the modulations and a bootstrapped target.
    Proposed in 'Learning Large-scale Neural Fields via Context Pruned Meta-Learning' by Jihoon Tack, et al. (2023)

    Everything is implemented to use this bootstrap correction. It is however NOT USED IN OUR PAPER.
    """
    updated_modulation = modulations_bootstrapped - modulations
    updated_modulation = updated_modulation.view(bs, -1)
    modulation_norm = torch.mean(updated_modulation ** 2, dim=-1)
    return modulation_norm

def modulation_consistency_firstframe(vdim, bs):
    """
    A function that calculates the L2-distance between the modulations and a bootstrapped target.
    Proposed in 'Learning Large-scale Neural Fields via Context Pruned Meta-Learning' by Jihoon Tack, et al. (2023)

    Everything is implemented to use this bootstrap correction. It is however NOT USED IN OUR PAPER.
    """
    print('vdim', vdim.shape)
    updated_v = vdim - vdim[0]   #subtract the first frame
    print('diff', updated_v.max())
    updated_v = updated_v.view(bs, -1)
    v_norm = torch.mean(updated_v ** 2, dim=-1)
    return v_norm


def get_grad_norm(grads, detach=True):
    grad_norm_list = []
    for grad in grads:
        if grad is None:
            grad_norm = 0
        else:
            if detach:
                grad_norm = torch.norm(grad.data, p=2, keepdim=True).unsqueeze(dim=0)
            else:
                grad_norm = torch.norm(grad, p=2, keepdim=True).unsqueeze(dim=0)

        grad_norm_list.append(grad_norm)
    return torch.norm(torch.cat(grad_norm_list, dim=0), p=2, dim=1)


def train_step(args, step, model_wrapper, optimizer, Data,  metric_logger, logger):
    """
    Function that performs a single meta update
    """
    criterion_guidance = torch.nn.MSELoss()
    model_wrapper.model.train()
    model_wrapper.coord_init()  # Reset coordinates
    model_wrapper.model.reset_modulations()  # Reset modulations (zero-initialization)
    if args.v_dim >0:
        model_wrapper.model.reset_vdim()
    if args.mode == 'time':
        data = Data['vid']
        time = Data['time']
    elif args.guidance == True:
        data = Data['vid']
        label = Data['label']
        batch_size = data.size(0)
    elif args.comment == 'separate2':
        batch_size = Data['vid'].size(0)
        data = Data
    else: 
        data = Data

   
    

        batch_size = data.size(0)
    
    
    if step % args.print_step == 0:
         learned_init = model_wrapper()
         input = data#['vid']

    """ Inner-loop optimization for G steps """
    if args.mode == 'time':
        loss_in = inner_adapt_time(model_wrapper=model_wrapper, data=data, time = time, step_size=args.inner_lr,
                            num_steps=args.inner_steps, first_order=False, sample_type=args.sample_type)
    elif args.mode == 'vae':
        loss_in = inner_adapt_vae(model_wrapper=model_wrapper, data=data,  step_size=args.inner_lr,
                            num_steps=args.inner_steps, first_order=False, sample_type=args.sample_type)
                                                
    elif args.v_dim ==0:
            loss_in = inner_adapt(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                            num_steps=args.inner_steps, first_order=False, sample_type=args.sample_type)
    else:
            loss_in = inner_adapt_v2(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr,
                            num_steps=args.inner_steps, first_order=False, sample_type=args.sample_type)
            modulations = model_wrapper.model.modulations.detach()
            v1 = model_wrapper.model.vdim.detach()
           
            


   
    """ Compute reconstruction loss using full context set"""
    model_wrapper.coord_init()
  #  modulations = model_wrapper.model.modulations.clone()  # Store modulations for consistency loss (not used)
    if args.v_dim > 0:
        vdim = model_wrapper.model.vdim.clone()
    if args.mode == 'time':
        loss_out = model_wrapper(data,time)  # Compute reconstruction loss
    else:
           loss_out = model_wrapper(data) 
    if step % args.print_step == 0:
        images = model_wrapper()  # Sample images

    """ Bootstrap correction for additional steps (NOT USED IN THIS PAPER) """
    _ = inner_adapt(model_wrapper=model_wrapper, data=data, step_size=args.inner_lr_boot,
                    num_steps=args.inner_steps_boot, first_order=True)
    #modulations_bootstrapped = model_wrapper.model.modulations.detach()
    if step % args.print_step == 0:
        target_boot = model_wrapper()


    """ Classification guidance if guidance is set to TRUE"""
    if args.guidance == True:
        target_pred = model_wrapper.model.simplecls(model_wrapper.model.modulations[None,...])
        loss_guidance = criterion_guidance(target_pred.float(), label[None,...].float() )
    
    """ Adversarial attack if adversarial is set to TRUE"""

  
    loss_boot = 0 * loss_out

    """ Modulation consistency loss between the different v of a batch """
  
  #  loss_boot2 = modulation_consistency_firstframe(vdim,  bs=batch_size)
  #  print('loss_boot2', loss_boot2)
    if args.guidance == True:

        loss_boot_weighted = args.lam * loss_guidance
    elif args.adversarial == True:
        loss_boot_weighted = args.lam * loss_boot


    else: 
        loss_boot_weighted = 0 * loss_out

    loss = loss_out.mean() + loss_boot_weighted.mean()
    psnro=psnr(loss_out.mean())
 
  

    """ Meta update (optimize shared weights) """
    optimizer.zero_grad()
   # print('loss meta', loss)
    torch.autograd.set_detect_anomaly(True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_wrapper.model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    

    """ Track stats"""
    metric_logger.meters['loss_inner'].update(loss_in.mean().item(), n=batch_size)
    metric_logger.meters['loss_outer'].update(loss_out.mean().item(), n=batch_size)
    metric_logger.meters['psnr_inner'].update(psnr(loss_in).mean().item(), n=batch_size)
    metric_logger.meters['psnr_outer'].update(psnr(loss_out).mean().item(), n=batch_size)
    metric_logger.meters['loss_boot'].update(loss_boot_weighted.mean().item(), n=batch_size)
    metric_logger.synchronize_between_processes()

    if step % args.print_step == 0:
        logger.scalar_summary('train/loss_inner', metric_logger.loss_inner.global_avg, step)
        logger.scalar_summary('train/loss_outer', metric_logger.loss_outer.global_avg, step)
        logger.scalar_summary('train/psnr_inner', metric_logger.psnr_inner.global_avg, step)
        logger.scalar_summary('train/psnr_outer', metric_logger.psnr_outer.global_avg, step)
        logger.scalar_summary('train/loss_boot', metric_logger.loss_boot.global_avg, step)
        logger.log_image('train/img_in', input, step)
        logger.log_image('train/learninit', learned_init, step)
        logger.log_image('train/img_inner', images, step)
        logger.log_image('train/img_bst', target_boot, step)

        logger.log('[TRAIN] [Step %3d] [LossInner %f] [LossOuter %f] [PSNRInner %.3f] [PSNROuter %.3f]' %
                   (step, metric_logger.loss_inner.global_avg, metric_logger.loss_outer.global_avg,
                    metric_logger.psnr_inner.global_avg, metric_logger.psnr_outer.global_avg))

    metric_logger.reset()


def inner_adapt(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none'):
    loss = 0.  # Initialize outer_loop loss

    """ Perform num_step (G) inner-loop updates """
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)  # Sample coordinates for the training step
        loss = inner_loop_step(model_wrapper, data, step_size, first_order)
      #  print('inner loss', loss)
    return loss

def inner_adapt_vae(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none'):
    loss = 0.  # Initialize outer_loop loss

    """ Perform num_step (G) inner-loop updates """
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)  # Sample coordinates for the training step
        loss = inner_loop_step_vae(model_wrapper, data, step_size, first_order)
      #  print('inner loss', loss)
    return loss

def inner_adapt_time(model_wrapper, data, time, step_size=1e-2, num_steps=3, first_order=False, sample_type='none'):
    loss = 0.  # Initialize outer_loop loss
    optimizer_time = optim.SGD(model_wrapper.model.time_layer.parameters(), lr=step_size)
    pytorch_total_params = sum(p.numel() for p in model_wrapper.model.time_layer.parameters())
    #print('numparams', pytorch_total_params)
    """ Perform num_step (G) inner-loop updates """
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)  # Sample coordinates for the training step

        loss = inner_loop_step_time(model_wrapper, data, time, step_size, first_order, optimizer_time)
    return loss
   



def inner_adapt_v2(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none'):
    loss = 0.  # Initialize outer_loop loss

    """ Perform num_step (G) inner-loop updates """
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)  # Sample coordinates for the training step
        loss = inner_loop_step_v2(model_wrapper, data, step_size, first_order)
    return loss

def inner_adapt_v3(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none'):
    loss = 0.  # Initialize outer_loop loss

    """ Perform num_step (G) inner-loop updates subsequent, first v then m """
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)  # Sample coordinates for the training step
        loss = inner_loop_step_v(model_wrapper, data, step_size, first_order)
      #  print('inner loss', loss)
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)  # Sample coordinates for the training step
        loss = inner_loop_step(model_wrapper, data, step_size, first_order)
      #  print('inner loss', loss)
    return loss

def inner_loop_step(model_wrapper, data, inner_lr=1e-2, first_order=False):
    batch_size = data.size(0)

    with torch.enable_grad():
        loss = model_wrapper(data)
        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.modulations,
            create_graph=not first_order,
        )[0]
        model_wrapper.model.modulations = model_wrapper.model.modulations - inner_lr * grads
    return loss

def inner_loop_step_vae(model_wrapper, data, inner_lr=1e-2, first_order=False):
    batch_size = data.size(0)

    with torch.enable_grad():
        loss = model_wrapper(data)
        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.z,
            allow_unused=True,
            create_graph=not first_order,
        )[0]
        model_wrapper.model.z = model_wrapper.model.z - inner_lr * grads
   
        # grads2 = torch.autograd.grad(
        #     loss.mean() * batch_size,
        #     model_wrapper.model.modulations_std, allow_unused=True,
        #     create_graph=not first_order,
        # )[0]
        # model_wrapper.model.modulations_std = model_wrapper.model.modulations_std - inner_lr * grads2
    return loss






def inner_loop_step_time(model_wrapper, data, time, inner_lr=1e-2, first_order=False, optimizer_time = None):
    batch_size = data.size(0)
    optimizer_time.zero_grad()
    with torch.enable_grad():
        loss = model_wrapper(data, time)
        loss.mean().backward(retain_graph = True)
        optimizer_time.step()
        # grads = torch.autograd.grad(
        #     loss.mean() * batch_size,
        #     model_wrapper.model.time_layer,
        #     create_graph=not first_order,
        # )[0]
        # gradients_weight = model_wrapper.model.time_layer.weight.grad
        # gradients_bias = model_wrapper.model.time_layer.bias.grad
        # print('weights', gradients_bias.max(), gradients_weight.max())
        # gradients_weight = torch.nn.Parameter(gradients_weight)
        # gradients_bias = torch.nn.Parameter(gradients_bias)

        # model_wrapper.model.time_layer.weight = model_wrapper.model.time_layer.weight - inner_lr * gradients_weight
        # model_wrapper.model.time_layer.bias = model_wrapper.model.time_layer.bias- inner_lr * gradients_bias
    return loss 

def inner_loop_step_v(model_wrapper, data, inner_lr=1e-2, first_order=False):
    batch_size = data.size(0)

    with torch.enable_grad():
        loss = model_wrapper(data)
        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.vdim,
            create_graph=not first_order,
        )[0]
        model_wrapper.model.vdim = model_wrapper.model.vdim - inner_lr * grads
    return loss


def inner_loop_step_v2(model_wrapper, data, inner_lr=1e-2, first_order=False):
  #  batch_size = data['vid'].size(0)
    batch_size = data.size(0)
    with torch.enable_grad():

        loss = model_wrapper(data)
        grads_v = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.vdim,
            allow_unused=True,
          #  retain_graph=True,
            create_graph=not first_order,
        )[0]
        model_wrapper.model.vdim = model_wrapper.model.vdim - inner_lr * grads_v
       

       # loss = model_wrapper(data)
        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.modulations,
            create_graph=not first_order,
        )[0]
        model_wrapper.model.modulations = model_wrapper.model.modulations - inner_lr * grads
    return loss


def inner_adapt_test_scale(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none',
                           scale_type='grad'):

    loss = 0.  # Initialize outer_loop loss
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)
       # if args.v_dim == 0:
         #   loss = inner_loop_step_tt_gradscale(model_wrapper, data, step_size, first_order, scale_type)
      #  else: 
            loss = inner_loop_step_tt_gradscale(model_wrapper, data, step_size, first_order, scale_type)

    return loss

def inner_adapt_test_scale_time(model_wrapper, Data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none',
                           scale_type='grad'):
    loss = 0.  # Initialize outer_loop loss
    data = Data['vid']
    time = Data['time']
   # optimizer_time = optim.Adam(model_wrapper.model.time_layer.parameters(), lr=0.01)
    optimizer_time = optim.SGD(model_wrapper.model.time_layer.parameters(), lr=step_size)

    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)
       # if args.v_dim == 0:
         #   loss = inner_loop_step_tt_gradscale(model_wrapper, data, step_size, first_order, scale_type)
      #  else: 
            loss = inner_loop_step_tt_gradscale_time(model_wrapper, data, time, step_size, first_order, scale_type, optimizer_time)

    return loss

def inner_adapt_test_scale_v2(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none',
                           scale_type='grad'):
    loss = 0.  # Initialize outer_loop loss
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)
       # if args.v_dim == 0:
         #   loss = inner_loop_step_tt_gradscale(model_wrapper, data, step_size, first_order, scale_type)
      #  else: 
            loss = inner_loop_step_tt_gradscale_v2(model_wrapper, data, step_size, first_order, scale_type)

    return loss

def inner_adapt_test_scale_vae(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none',
                           scale_type='grad'):
    loss = 0.  # Initialize outer_loop loss
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)
            loss = inner_loop_step_tt_gradscale_vae(model_wrapper, data, step_size, first_order, scale_type)

    return loss

def inner_adapt_test_scale_v3(model_wrapper, data, step_size=1e-2, num_steps=3, first_order=False, sample_type='none',
                           scale_type='grad'):
    loss = 0.  # Initialize outer_loop loss
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)
            loss = inner_loop_step_tt_gradscale_v(model_wrapper, data, step_size, first_order, scale_type)
    for step_inner in range(num_steps):
        if sample_type != 'none':
            model_wrapper.sample_coordinates(sample_type, data)
            loss = inner_loop_step_tt_gradscale(model_wrapper, data, step_size, first_order, scale_type)

    return loss


def inner_loop_step_tt_gradscale_time(model_wrapper, data, time, inner_lr=1e-2, first_order=False, scale_type='grad', optimizer_time=None):
    batch_size = data.size(0)
    optimizer_time.zero_grad()

  #  with torch.enable_grad():
    #     subsample_loss = model_wrapper(data,time)
      #   subsample_loss.mean().backward()
      #   grad_norm = np.sqrt(sum(torch.norm(p.grad)**2 for p in model_wrapper.model.time_layer.parameters() if p.grad is not None))
        # subsample_grad = torch.autograd.grad(
    #         subsample_loss.mean() * batch_size,
    #         model_wrapper.model.modulations,
    #         create_graph=False,
    #         allow_unused=True
    #     )[0]

    model_wrapper.model.zero_grad()
    model_wrapper.coord_init()

    with torch.enable_grad():
        loss = model_wrapper(data, time)
        loss.mean().backward()
        optimizer_time.step()

    # if scale_type == 'grad':
    #     # Gradient rescaling at test-time
    #     subsample_grad_norm = get_grad_norm(subsample_grad, detach=True)
    #     grad_norm = get_grad_norm(grads, detach=True)
    #     grad_scale = subsample_grad_norm / (grad_norm + 1e-16)
    #     grad_scale_ = grad_scale.view((batch_size,) + (1,) * (len(grads.shape) - 1)).detach()
    # else:
    #     raise NotImplementedError()
   
    return loss



def inner_loop_step_tt_gradscale(model_wrapper, data, inner_lr=1e-2, first_order=False, scale_type='grad'):
  #  batch_size = data['vid'].size(0)
    batch_size = data.size(0)
    model_wrapper.model.zero_grad()
    with torch.enable_grad():
        subsample_loss = model_wrapper(data)
        subsample_grad = torch.autograd.grad(
            subsample_loss.mean() * batch_size,
            model_wrapper.model.modulations,
            create_graph=False,
            allow_unused=True
        )[0]

    model_wrapper.model.zero_grad()
    model_wrapper.coord_init()

    with torch.enable_grad():
        loss = model_wrapper(data)

        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.modulations,
            create_graph=not first_order,
            allow_unused=True
        )[0]

    if scale_type == 'grad':
        # Gradient rescaling at test-time
        subsample_grad_norm = get_grad_norm(subsample_grad, detach=True)
        grad_norm = get_grad_norm(grads, detach=True)
        grad_scale = subsample_grad_norm / (grad_norm + 1e-16)
        grad_scale_ = grad_scale.view((batch_size,) + (1,) * (len(grads.shape) - 1)).detach()
    else:
        raise NotImplementedError()
    model_wrapper.model.modulations = model_wrapper.model.modulations - inner_lr *grads* grad_scale_ 

    return loss


def inner_loop_step_tt_gradscale_v(model_wrapper, data, inner_lr=1e-2, first_order=False, scale_type='grad'):
    batch_size = data.size(0)
    model_wrapper.model.zero_grad()

    with torch.enable_grad():
        subsample_loss = model_wrapper(data)
        subsample_grad = torch.autograd.grad(
            subsample_loss.mean() * batch_size,
            model_wrapper.model.vdim,
            create_graph=False,
            allow_unused=True
        )[0]

    model_wrapper.model.zero_grad()
    model_wrapper.coord_init()

    with torch.enable_grad():
        loss = model_wrapper(data)

        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.vdim,
            create_graph=not first_order,
            allow_unused=True
        )[0]
    if scale_type == 'grad':
        # Gradient rescaling at test-time
        subsample_grad_norm = get_grad_norm(subsample_grad, detach=True)
        grad_norm = get_grad_norm(grads, detach=True)
        grad_scale = subsample_grad_norm / (grad_norm + 1e-16)
        grad_scale_ = grad_scale.view((1,) + (1,) * (len(grads.shape) - 1)).detach()
    else:
        raise NotImplementedError()
    
    model_wrapper.model.vdim = model_wrapper.model.vdim - inner_lr *grads* grad_scale_ 

    return loss



def inner_loop_step_tt_gradscale_v2(model_wrapper, data, inner_lr=1e-2, first_order=False, scale_type='grad'):
   # batch_size = data['vid'].size(0)
    batch_size = data.size(0)
    model_wrapper.model.zero_grad()
    

    with torch.enable_grad():

        subsample_loss = model_wrapper(data)

        subsample_grad_v = torch.autograd.grad(
            subsample_loss.mean() * batch_size,
            model_wrapper.model.vdim,
            retain_graph=True,
            allow_unused=True
        )[0]

        subsample_grad = torch.autograd.grad(
            subsample_loss.mean() * batch_size,
            model_wrapper.model.modulations,
            create_graph=False,
            allow_unused=True
        )[0]

    model_wrapper.model.zero_grad()
    model_wrapper.coord_init()

    with torch.enable_grad():
        loss = model_wrapper(data)
        grads_v = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.vdim,
            retain_graph=True,
            allow_unused=True
        )[0]
        if scale_type == 'grad':
            # Gradient rescaling at test-time
            subsample_grad_norm = get_grad_norm(subsample_grad_v, detach=True)
            grad_norm = get_grad_norm(grads_v, detach=True)
            grad_scale = subsample_grad_norm / (grad_norm + 1e-16)
            grad_scale_ = grad_scale.view((1,) + (1,) * (len(grads_v.shape) - 1)).detach()
        else:
            raise NotImplementedError()
        model_wrapper.model.vdim = model_wrapper.model.vdim - inner_lr *grads_v* grad_scale_ 
        loss_m = model_wrapper(data)
        grads = torch.autograd.grad(
            loss_m.mean() * batch_size,
            model_wrapper.model.modulations,
            create_graph=not first_order,
            allow_unused=True
        )[0]
    if scale_type == 'grad':
        # Gradient rescaling at test-time
        subsample_grad_norm = get_grad_norm(subsample_grad, detach=True)
        grad_norm = get_grad_norm(grads, detach=True)
        grad_scale = subsample_grad_norm / (grad_norm + 1e-16)
        grad_scale_ = grad_scale.view((batch_size,) + (1,) * (len(grads.shape) - 1)).detach()
    else:
        raise NotImplementedError()

    model_wrapper.model.modulations = model_wrapper.model.modulations - inner_lr *grads* grad_scale_ 

    return loss_m




def inner_loop_step_tt_gradscale_vae(model_wrapper, data, inner_lr=1e-2, first_order=False, scale_type='grad'):
    batch_size = data.size(0)
    model_wrapper.model.zero_grad()
    

    with torch.enable_grad():


        subsample_loss = model_wrapper(data)

        subsample_grad_mean = torch.autograd.grad(
            subsample_loss.mean() * batch_size,
            model_wrapper.model.z,
            retain_graph=True,
            allow_unused=True
        )[0]

        # subsample_grad_std = torch.autograd.grad(
        #     subsample_loss.mean() * batch_size,
        #     model_wrapper.model.modulations_std,
        #     create_graph=False,
        #     allow_unused=True
        # )[0]

    model_wrapper.model.zero_grad()
    model_wrapper.coord_init()

    with torch.enable_grad():
        loss = model_wrapper(data)
        grads = torch.autograd.grad(
            loss.mean() * batch_size,
            model_wrapper.model.z,
            retain_graph=True,
            allow_unused=True
        )[0]
        if scale_type == 'grad':
            # Gradient rescaling at test-time
            subsample_grad_norm = get_grad_norm(subsample_grad_mean, detach=True)
            grad_norm = get_grad_norm(grads, detach=True)
            grad_scale = subsample_grad_norm / (grad_norm + 1e-16)
            grad_scale_ = grad_scale.view((batch_size,) + (1,) * (len(grads.shape) - 1)).detach()
        else:
            raise NotImplementedError()
        model_wrapper.model.z = model_wrapper.model.z - inner_lr *grads* grad_scale_ 

       
        # grads2 = torch.autograd.grad(
        #     loss.mean() * batch_size,
        #     model_wrapper.model.modulations_std,
        #     create_graph=not first_order,
        #     allow_unused=True
        # )[0]
    # if scale_type == 'grad':
    #     # Gradient rescaling at test-time
    #     subsample_grad_norm = get_grad_norm(subsample_grad_std, detach=True)
    #     grad_norm = get_grad_norm(grads2, detach=True)
    #     grad_scale = subsample_grad_norm / (grad_norm + 1e-16)
    #     grad_scale_ = grad_scale.view((batch_size,) + (1,) * (len(grads2.shape) - 1)).detach()
    # else:
    #     raise NotImplementedError()

   # model_wrapper.model.modulations_std = model_wrapper.model.modulations_std - inner_lr *grads2* grad_scale_ 

    return loss