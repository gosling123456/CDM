import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir='./sample_npz/SAR1-18_fb_bs8')

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    msg = model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    print(msg)
    model.to(dist_util.dev())
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    
    all_samples = []
    
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, all_sample = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        
        all_sample = ((all_sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        all_sample = all_sample.permute(0, 1, 3, 4, 2)
        all_sample = all_sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        
        # all_gathered_samples = [th.zeros_like(all_sample) for _ in range(dist.get_world_size())]
        # print(len(all_gathered_samples),all_gathered_samples[0].shape)
        # dist.all_gather(all_gathered_samples, all_sample)  # gather not supported with NCCL
        # print(all_sample.cpu().numpy().shape)
        all_samples.extend([all_sample.cpu().numpy()])
        
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    
    # print(len(all_sample), all_samples[1].shape)
    all_arr = np.concatenate(all_samples, axis=0)
    all_arr = all_arr[: args.num_samples]
    print(all_arr.shape)
    
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        
        all_shape_str = "x".join([str(x) for x in all_arr.shape])
        all_out_path = os.path.join(logger.get_dir(), f"all_samples_{all_shape_str}.npz")
        
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)
            np.savez(all_out_path, all_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=100,
        batch_size=2,
        use_ddim=False,
        model_path="../下游任务/data/dz/diffusion_model/IDDPM_CSSL/model_log/SAR1-18_fb_bs8/ema_0.9999_250000.pt",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
