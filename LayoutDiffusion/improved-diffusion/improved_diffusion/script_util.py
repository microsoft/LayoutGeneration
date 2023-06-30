import argparse
import inspect

from . import gaussian_diffusion as gd
from .discrete_diffusion import DiffusionTransformer
from .respace import SpacedDiffusion, space_timesteps
from .transformer_model import TransformerModel,DiscreteTransformerModel
NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
 
    return dict(
        seq_length=121,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        model_arch='transformer',
        in_channel=8,
        out_channel=8,
        training_mode='emb',
        vocab_size=66,
        config_name='bert-base-uncased',
        experiment_mode='lm',
        logits_mode=1,
        ungen=False,
        constrained=None,
        self_cond=False,
        att_1=0.99999,
        alignment_loss=False,
        alignment_weight=1e2,
        aux_loss=True,
        aux_loss_weight=1e-3,
    )
    


def create_model_and_diffusion(
    seq_length,
    class_cond,
    learn_sigma,
    sigma_small,
    num_channels,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    model_arch,
    in_channel,
    out_channel,
    training_mode,
    vocab_size,
    config_name,
    experiment_mode,
    logits_mode,
    ungen,
    constrained,
    self_cond,
    att_1,
    alignment_loss,
    alignment_weight,
    aux_loss,
    aux_loss_weight,
    **kwargs,
):  
    layout=0
    matrix_policy=0,
    if training_mode=='rico':
        training_mode='e2e'
        layout=1
    if training_mode=='discrete1':
        training_mode='discrete'
        matrix_policy=1
    model = create_model(
        seq_length,
        num_channels,
        num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        model_arch=model_arch,
        in_channel=in_channel,
        out_channel=out_channel,
        training_mode=training_mode,
        vocab_size=vocab_size,
        config_name=config_name,
        experiment_mode=experiment_mode,
        logits_mode=logits_mode,
        layout=layout,
        ungen=ungen,
        predict_xstart=predict_xstart,
        constrained=constrained,
        self_cond=self_cond,
        matrix_policy=matrix_policy,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
        model_arch=model_arch,
        training_mode=training_mode,
        ungen=ungen,
        self_cond=self_cond,
        matrix_policy=matrix_policy,
        att_1=att_1,
        vocab_size=vocab_size,
        seq_len=seq_length,
        alignment_loss=alignment_loss,
        alignment_weight=alignment_weight,
        aux_loss=aux_loss,
        aux_loss_weight=aux_loss_weight,
    )

    return model, diffusion


def create_model(
    seq_length,
    num_channels,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    model_arch,
    in_channel=8,
    out_channel=8,
    training_mode='emb',
    vocab_size=None,
    config_name='',
    experiment_mode='lm',
    logits_mode=1,
    layout=0,
    ungen=False,
    predict_xstart=False,
    constrained=None,
    self_cond=False,
    matrix_policy=0,
):
    if model_arch == 'transformer':

        # for discrete diffusion #

        if training_mode=='discrete': 
            return DiscreteTransformerModel(
                in_channels=in_channel,  # 3, DEBUG**
                model_channels=num_channels,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                num_classes=(NUM_CLASSES if class_cond else None),
                use_checkpoint=use_checkpoint,
                config_name=config_name,
                training_mode=training_mode,
                vocab_size=vocab_size,
                constrained=constrained,
                self_cond=self_cond,
                matrix_policy=matrix_policy,
            )


        ### below for diffusion-lm ###


        if not layout: #seq input
            if training_mode!='bit':
                return TransformerModel(
                    in_channels=in_channel,  # 3, DEBUG**
                    model_channels=num_channels,
                    out_channels=(out_channel if not learn_sigma else out_channel*2),  # DEBUG**  (3 if not learn_sigma else 6),
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    num_classes=(NUM_CLASSES if class_cond else None),
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_heads_upsample=num_heads_upsample,
                    use_scale_shift_norm=use_scale_shift_norm,
                    config_name=config_name,
                    training_mode=training_mode,
                    vocab_size=vocab_size,
                    experiment_mode=experiment_mode,
                    logits_mode=logits_mode,
                    layout=layout,
                    constrained=constrained,
                    self_cond=self_cond,
                )
            
    else:
        raise NotImplementedError


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
    model_arch='transformer',
    training_mode='emb',
    ungen=False,
    self_cond=False,
    matrix_policy=0,
    att_1=0.9,
    vocab_size=159,
    seq_len=121,
    alignment_loss=True,
    alignment_weight=1e5,
    aux_loss=True,
    aux_loss_weight=1e-3,
):
    

    if training_mode == 'discrete': 
        
        # for discrete diffusion #

        return DiffusionTransformer(
            diffusion_step=steps,
            matrix_policy=matrix_policy,
            att_1=att_1,
            alpha_init_type=noise_schedule,
            num_classes=vocab_size,
            adaptive_auxiliary_loss=aux_loss,
            auxiliary_loss_weight=aux_loss_weight,
            rescale_weight=rescale_timesteps,
            content_seq_len=seq_len,
            alignment_loss=alignment_loss,
            alignment_weight=alignment_weight,
        )
    else:
        betas = gd.get_named_beta_schedule(noise_schedule, steps)
    

    ### below for diffusion-lm ###


    if training_mode == 'e2e':
        # end to end training
        if use_kl:
            loss_type = gd.LossType.E2E_KL
        else:
            loss_type = gd.LossType.E2E_MSE
    elif training_mode == 'e2e-simple':
        if use_kl:
            loss_type = gd.LossType.E2E_Simple_KL
        else:
            loss_type = gd.LossType.E2E_Simple_MSE

    else:
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        model_arch=model_arch,
        training_mode=training_mode,
        ungen=ungen,
        self_cond=self_cond,
    )


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
