"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os, json
import pickle
import numpy as np
import torch as th
import torch.distributed as dist
from transformers import set_seed
from improved_diffusion.rounding import rounding_func, load_models, load_tokenizer

from improved_diffusion.test_util import get_weights, denoised_fn_round

from improved_diffusion import dist_util, logger
from functools import partial
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from einops import rearrange, reduce

def int2bit(x, n=7):
    # Convert integers into the corresponding binary bits.
    device=x.device
    x = th.bitwise_right_shift(x.unsqueeze(-1).int(), th.range(0,n,dtype=th.int64,device=device))
    x = th.fmod(x, 2).float()
    return x*2-1

def bit2int(x):
    device=x.device
    # Convert binary bits into the corresponding integers.
    x = (x+1).int()
    x=th.where(x>1,th.ones_like(x,device=device),x)
    n = x.shape[-1]
    x = th.sum(x * (2 ** th.range(0,n-1,device=device)), -1)
    return x

def main():
    set_seed(101)

    args = create_argparser().parse_args()

    constrained=args.constrained # coupled with line69
    
    dist_util.setup_dist()
    logger.configure()

    # load configurations.
    config_path = os.path.join(os.path.split(args.model_path)[0], "training_args.json")
    with open(config_path, 'rb', ) as f:
        training_args = json.load(f)
    training_args['batch_size'] = args.batch_size
    args.__dict__.update(training_args)
    args.sigma_small = True

    args.constrained=constrained
    if args.constrained is not None and args.constrained.startswith("refine"):
        args.e2e_train=args.e2e_train+'_'+args.constrained
        args.constrained='refine'

    if args.experiment == 'random1': args.experiment = 'random'
    # if args.training_mode=='discrete': args.training_mode='discrete1'
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.log(f'the parameter count is {pytorch_total_params}')

    model.to(dist_util.dev())
    model.eval() # DEBUG
    
    if args.training_mode=='discrete1':
        args.training_mode='discrete'

    if args.constrained is not None:
        from improved_diffusion.text_datasets import load_data_text
        model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                        os.path.split(args.model_path)[0])
        print('conditional generation mode --> load data')
        rev_tokenizer = {v: k for k, v in tokenizer.items()}

        data = load_data_text(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            class_cond=args.class_cond,
            data_args=args,
            model=model2,
            deterministic=True,
            task_mode=args.modality,
            padding_mode=args.padding_mode,  # block, pad
            split='test',
            load_vocab=rev_tokenizer,
        )


    if args.training_mode=='rico' and not args.ungen:
        from improved_diffusion.text_datasets import load_data_text
        model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                        os.path.split(args.model_path)[0])
        print('conditional generation mode --> load data')
        rev_tokenizer = {v: k for k, v in tokenizer.items()}
        data = load_data_text(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            class_cond=args.class_cond,
            data_args=args,
            model=model2,
            deterministic=True,
            task_mode=args.modality,
            padding_mode=args.padding_mode,  # block, pad
            split='test',
            load_vocab=rev_tokenizer,
        )

    model2, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                    os.path.split(args.model_path)[0])
    if args.training_mode.startswith('e2e'):
        model2.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

    logger.log("sampling...")
    all_images = []
    all_labels = []
    model3 = get_weights(model2, args)


    while len(all_images) * args.batch_size < args.num_samples:

        ### prepare for the sampling ###
        model_kwargs = {}
        if args.constrained is not None:
            batch, model_kwargs = next(data)

            model_kwargs["y"]=model_kwargs.pop('input_ids').to(dist_util.dev())
            
        # rico here
        if args.training_mode=='rico' and not args.ungen:
            batch, types = next(data)
            model_kwargs["y"]=types['input_ids'].to(dist_util.dev())
            if 'pad_mask' in types.keys():
                model_kwargs["src_mask"]=types['pad_mask'].to(dist_util.dev())
            all_labels.append(types['input_ids'])

        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        
        sample_fn = (
            diffusion.sample_fast if args.training_mode=='discrete' else diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        # prepare sample shape
        if args.mbr_sample > 1 and args.experiment_mode == 'conditional_gen':
            sample_shape = (args.batch_size * args.mbr_sample, args.seq_length, args.in_channel)
        elif args.training_mode=='rico' and not args.ungen:
            sample_shape = (args.batch_size, 20, 4)
            args.clamp=None
        elif args.training_mode=='rico' and args.ungen:
            sample_shape=(args.batch_size,20,4+args.in_channel)
        elif args.training_mode=='bit':
            sample_shape=(args.batch_size,args.seq_length,8)
        elif args.training_mode=='discrete':
            sample_shape=(args.batch_size,args.seq_length)
        else:
            sample_shape = (args.batch_size, args.seq_length, args.in_channel)

        if args.training_mode != 'discrete':
            ### start sampling ###
            sample = sample_fn(
                model,
                sample_shape,
                noise=(model.get_embeds(model_kwargs['y']) if args.constrained=='refine' else None),
                clip_denoised=args.clip_denoised,
                denoised_fn=partial(denoised_fn_round, args, model3.cuda()) if args.clamp == 'clamp' else None,
                model_kwargs=model_kwargs,
                top_p =args.top_p,
                multistep=args.multistep   #junyi added
            )

            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            if args.class_cond:
                gathered_labels = [
                    th.zeros_like(classes) for _ in range(dist.get_world_size())
                ]
                dist.all_gather(gathered_labels, classes)
                all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
            logger.log(f"created {len(all_images) * args.batch_size} samples")
        else:
            T_refine = 50 if args.vocab_size == 159 else 60
            sample = sample_fn(
                model,
                sample_shape,
                sample_start_step=(T_refine if args.constrained=='refine' else args.diffusion_steps),
                content_token=(model_kwargs['y'] if args.constrained is not None else None),
                multistep=args.multistep,
                constrained=args.constrained,
            )
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
            logger.log(f"created {len(all_images) * args.batch_size} samples")

    arrs = np.concatenate(all_images, axis=0)

    ### gather output ###
    if not args.multistep:
        arrs = arrs[: args.num_samples * args.mbr_sample]
        arrs=th.unsqueeze(th.from_numpy(arrs),0).numpy()
        
    else:
        if args.training_mode != 'discrete':
            arrs=th.from_numpy(arrs).permute(1,0,2,3).numpy()
        else:
            arrs=th.from_numpy(arrs).permute(1,0,2).numpy()
        arrs = arrs[: args.num_samples * args.mbr_sample]

    for idx,arr in enumerate(arrs):
        ## post process starts here
        if args.training_mode!='rico' and args.training_mode!='discrete' and diffusion.training_mode.startswith('e2e'): #normal rico seq
            word_lst_e2e = []
            x_t = th.tensor(arr).cuda()
            if args.model_arch == 'conv-unet':
                reshaped_x_t = x_t.view(x_t.size(0), -1, x_t.size(-1))
            else:
                reshaped_x_t = x_t
            logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
            cands = th.topk(logits, k=1, dim=-1)
            sample = cands.indices
            tokenizer = load_tokenizer(args.modality, args.experiment, os.path.split(args.model_path)[0])
            for seq in cands.indices:
                if isinstance(tokenizer, dict):
                    tokens = " ".join([tokenizer[x[0].item()] for x in seq])
                else:
                    tokens = tokenizer.decode(seq.squeeze(-1))
                word_lst_e2e.append(tokens)
        
        if args.ungen:
            all_labels = []
            x_t = th.tensor(arr).cuda()
            bbox=x_t[:,:,:4]
            reshaped_x_t = x_t[:,:,4:]
            logits = model.get_logits(reshaped_x_t)  # bsz, seqlen, vocab
            cands = th.topk(logits, k=1, dim=-1)
            sample = cands.indices
            all_labels.append(sample.squeeze(-1).cpu())
            arr=bbox.cpu().numpy()
        

        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
            label_arr = label_arr[: args.num_samples]
        if dist.get_rank() == 0:
            shape_str = "x".join([str(x) for x in arr.shape])
            model_base_name = os.path.basename(os.path.split(args.model_path)[0]) + f'.{os.path.split(args.model_path)[1]}'
            out_path = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}.npz")
            # out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
            logger.log(f"saving to {out_path}")
            if args.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)
        ## pose process ends here

        dist.barrier()
        logger.log("sampling complete")

        ## to save result ##
        if args.verbose == 'yes':
            logger.log('decode by rounding. ')
            if (args.training_mode!='rico' and args.training_mode!='discrete' and diffusion.training_mode.startswith('e2e')): ## normal rico seq
                word_lst = word_lst_e2e
            
            elif args.training_mode=='rico':
                labels=th.cat(all_labels,dim=0)
                outputs=[]
                
                for i in range(arr.shape[0]):
                    output={}
                    output['pred']=[]
                    label_single=labels[i]
                    bbox_single=th.from_numpy(arr[i])
                    mask=th.nonzero(label_single).squeeze(-1)
                    if args.multistep:
                        mask=th.arange(label_single.shape[0])
                    output['pred'].append(th.index_select(bbox_single, 0, mask))
                    output['pred'].append(th.index_select(label_single, 0, mask))
                    outputs.append(output)

            
            elif args.training_mode=='bit':
                set_seed(101)
                model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                            os.path.split(args.model_path)[0])
                word_lst=[]
                for arr_single in arr:
                    
                    text_emb = th.tensor(arr_single)
                    if len(text_emb.shape) > 2:
                        text_emb = text_emb.view(-1, text_emb.size(-1))
                    else:
                        text_emb = text_emb
                    indices=bit2int(text_emb)
                    indices=th.where(indices<158,indices,th.zeros_like(indices))
                    decoded_out = " ".join([tokenizer[int(i+1)] for i in indices.tolist()])
                    word_lst.append(decoded_out)
            
            elif args.training_mode=='discrete':
                set_seed(101)
                word_lst=[]
                model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                            os.path.split(args.model_path)[0])
                for arr_single in arr:

                    text_emb = th.tensor(arr_single)
                    if len(text_emb.shape) > 2:
                        indices = text_emb.view(-1, text_emb.size(-1))
                    else:
                        indices = text_emb
                    tokenizer[args.vocab_size-1]='MASK'
                    decoded_out = " ".join([tokenizer[int(i)] for i in indices.tolist()])
                    word_lst.append(decoded_out)

            else: # no use
                set_seed(101)
                model, tokenizer = load_models(args.modality, args.experiment, args.model_name_or_path, args.in_channel,
                                            os.path.split(args.model_path)[0])
                word_lst = rounding_func(args.experiment, arr, model, tokenizer,
                                        emb_scale_factor=args.emb_scale_factor)
            
            if args.training_mode!='rico': #rico seq
                out_path2 = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}_elem{str(idx+1)}.txt")
                fout = open(out_path2, 'w')
                for (xx) in zip( word_lst):
                    print(xx[0], file=fout)
                fout.close()
                print(f'written the decoded output to {out_path2}')

                out_path2 = os.path.join(args.out_dir, f"{model_base_name}.samples_{args.top_p}_elem{str(idx+1)}.json")
                fout = open(out_path2, 'w')
                for (xx) in zip(word_lst):
                    print(json.dumps(xx), file=fout)
                fout.close()
                print(f'written the decoded output to {out_path2}')
            else: #rico here
                if not args.multistep:
                    out_path2 = os.path.join(args.out_dir, "rico_raw.pt")
                else:
                    out_path2 = os.path.join(args.out_dir, f"rico_raw_elem{str(idx+1)}.pt")
                with open(out_path2,'wb') as f:
                    pickle.dump(outputs,f)

def create_argparser():
    defaults = dict(
        clip_denoised=False,
        num_samples=50,
        batch_size=64,
        use_ddim=False,
        mbr_sample=1,
        model_path="",
        model_arch='conv-unet',
        verbose='yes',
        out_dir="diffusion_lm/improved_diffusion/out_gen",
        multistep=True
    )
    text_defaults = dict(modality='text',
                         dataset_name='wikitext',
                         dataset_config_name='wikitext-2-raw-v1',
                         model_name_or_path='predictability/diff_models/compress_e=5_b=60_m=gpt2_wikitext-103-raw-v1_None',
                         experiment='gpt2_pre_compress', model_arch='trans-unet',
                         preprocessing_num_workers=1,
                         emb_scale_factor=1.0, top_p=-1., split='valid', clamp='clamp')
    defaults.update(model_and_diffusion_defaults())
    defaults.update(text_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
