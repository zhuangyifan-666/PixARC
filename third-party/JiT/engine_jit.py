import copy
import math
import os
import shutil
import sys

import cv2
import numpy as np
import torch
from torch_fidelity.metric_fid import fid_featuresdict_to_statistics, fid_statistics_to_metric
from torch_fidelity.metric_isc import isc_featuresdict_to_metric
from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_id_cached

import util.lr_sched as lr_sched
import util.misc as misc


def calculate_metrics_with_reference_stats(samples_path, statistics_path, batch_size=64, cuda=True, verbose=False):
    """Compute IS and FID with the upstream torch-fidelity package.

    Upstream torch-fidelity requires a second image input for FID and does not
    accept the custom ``fid_statistics_file`` argument used by the original JiT
    code. Extracting both requested Inception features in one pass preserves the
    same metric definitions while allowing the bundled reference statistics to
    be used directly.
    """
    feature_layer_isc = 'logits_unbiased'
    feature_layer_fid = '2048'
    metric_kwargs = {
        'input1': samples_path,
        'batch_size': batch_size,
        'cuda': cuda,
        'cache': False,
        'samples_shuffle': True,
        'rng_seed': 2020,
        'verbose': verbose,
    }
    feature_extractor = create_feature_extractor(
        'inception-v3-compat',
        [feature_layer_isc, feature_layer_fid],
        **metric_kwargs,
    )
    features_dict = extract_featuresdict_from_input_id_cached(
        1,
        feature_extractor,
        **metric_kwargs,
    )

    metrics = isc_featuresdict_to_metric(features_dict, feature_layer_isc, **metric_kwargs)
    generated_stats = fid_featuresdict_to_statistics(features_dict, feature_layer_fid)
    with np.load(statistics_path) as reference_file:
        reference_stats = {
            'mu': reference_file['mu'].astype(generated_stats['mu'].dtype, copy=False),
            'sigma': reference_file['sigma'].astype(generated_stats['sigma'].dtype, copy=False),
        }
    metrics.update(fid_statistics_to_metric(generated_stats, reference_stats, verbose))
    return metrics


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (x, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # per iteration (instead of per epoch) lr scheduler
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        # normalize image to [-1, 1]
        x = x.to(device, non_blocking=True).to(torch.float32).div_(255)
        x = x * 2.0 - 1.0
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            loss = model(x, labels)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

        if log_writer is not None:
            # Use epoch_1000x as the x-axis in TensorBoard to calibrate curves.
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if data_iter_step % args.log_freq == 0:
                log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
                log_writer.add_scalar('lr', lr, epoch_1000x)


def evaluate(model_without_ddp, args, epoch, batch_size=64, log_writer=None):

    model_without_ddp.eval()
    world_size = misc.get_world_size()
    local_rank = misc.get_rank()
    if args.num_images <= 0:
        raise ValueError('--num_images must be positive')
    if batch_size <= 0:
        raise ValueError('--gen_bsz must be positive')
    if not args.skip_metrics and args.metrics_batch_size <= 0:
        raise ValueError('--metrics_batch_size must be positive')
    if not args.skip_metrics and args.img_size not in (256, 512):
        raise ValueError('Pre-computed FID statistics are only available for 256 and 512 resolutions')
    class_num = args.class_num
    if class_num <= 0:
        raise ValueError('--class_num must be positive')
    if args.num_images % class_num != 0:
        raise ValueError('--num_images must be divisible by --class_num')
    num_steps = math.ceil(args.num_images / (batch_size * world_size))

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        args.output_dir,
        "{}-steps{}-cfg{}-interval{}-{}-image{}-res{}".format(
            model_without_ddp.method, model_without_ddp.steps, model_without_ddp.cfg_scale,
            model_without_ddp.cfg_interval[0], model_without_ddp.cfg_interval[1], args.num_images, args.img_size
        )
    )
    print("Save to:", save_folder)
    if misc.get_rank() == 0:
        os.makedirs(save_folder, exist_ok=True)
    torch.distributed.barrier()

    # switch to ema params, hard-coded to be the first one
    model_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    ema_state_dict = copy.deepcopy(model_without_ddp.state_dict())
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        assert name in ema_state_dict
        ema_state_dict[name] = model_without_ddp.ema_params1[i]
    print("Switch to ema")
    model_without_ddp.load_state_dict(ema_state_dict)

    # ensure that the number of images per class is equal.
    class_label_gen_world = np.arange(0, class_num, dtype=np.int64).repeat(args.num_images // class_num)
    num_padding_labels = num_steps * batch_size * world_size - args.num_images
    if num_padding_labels:
        class_label_gen_world = np.pad(class_label_gen_world, (0, num_padding_labels))

    for i in range(num_steps):
        print("Generation step {}/{}".format(i + 1, num_steps))

        start_idx = world_size * batch_size * i + local_rank * batch_size
        end_idx = start_idx + batch_size
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.from_numpy(labels_gen).long().cuda()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            sampled_images = model_without_ddp.generate(labels_gen)

        torch.distributed.barrier()

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        # distributed save images
        for b_id in range(sampled_images.size(0)):
            img_id = i * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip(sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            image_path = os.path.join(save_folder, '{}.png'.format(str(img_id).zfill(5)))
            if not cv2.imwrite(image_path, gen_img):
                raise OSError('Failed to save generated image at {}'.format(image_path))

    torch.distributed.barrier()

    # back to no ema
    print("Switch back from ema")
    model_without_ddp.load_state_dict(model_state_dict)
    del model_state_dict, ema_state_dict
    torch.cuda.empty_cache()

    # compute FID and IS
    if misc.is_main_process():
        if args.skip_metrics:
            print("Skipping FID and Inception Score; generated images were kept at", save_folder)
        else:
            fid_statistics_file = os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                'fid_stats',
                'jit_in{}_stats.npz'.format(args.img_size),
            )
            metrics_dict = calculate_metrics_with_reference_stats(
                samples_path=save_folder,
                statistics_path=fid_statistics_file,
                batch_size=args.metrics_batch_size,
                cuda=True,
                verbose=False,
            )
            fid = metrics_dict['frechet_inception_distance']
            inception_score = metrics_dict['inception_score_mean']
            postfix = "_cfg{}_res{}".format(model_without_ddp.cfg_scale, args.img_size)
            if log_writer is not None:
                log_writer.add_scalar('fid{}'.format(postfix), fid, epoch)
                log_writer.add_scalar('is{}'.format(postfix), inception_score, epoch)
            print("FID: {:.4f}, Inception Score: {:.4f}".format(fid, inception_score))
            if args.keep_generated_images:
                print("Generated images were kept at", save_folder)
            else:
                shutil.rmtree(save_folder)

    torch.distributed.barrier()
