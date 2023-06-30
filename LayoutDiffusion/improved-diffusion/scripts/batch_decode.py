import os, sys, glob

full_lst = glob.glob(sys.argv[1])
top_p = -1.0 if len(sys.argv) <= 2 else sys.argv[2]
pattern_ = "model" if len(sys.argv) <= 3 else sys.argv[3]
bz = 50 if len(sys.argv) <= 4 else sys.argv[4]
num_samples = 50 if len(sys.argv) <= 5 else sys.argv[5]
multistep = False if len(sys.argv) <= 6 else sys.argv[6]
ckpt = "-1" if len(sys.argv) <= 7 else sys.argv[7]
constrained = None if len(sys.argv) <= 8 else sys.argv[8]

modality = "e2e-tgt"


output_lst = []
for lst in full_lst:
    try:
        tgt = sorted(glob.glob(f"{lst}/{pattern_}*pt"))[int(ckpt)]
        lst = os.path.split(lst)[1]
        num = 1
    except:
        continue
    model_arch = "transformer"
    mode = "text"  # or '1d-unet' in model_arch_

    # diffusion_steps= 4000
    # noise_schedule = 'cosine'
    # dim = dim_.split('rand')[1]

    if "synth" in lst:
        modality = "synth"
    elif "pos" in lst:
        modality = "pos"
    elif "image" in lst:
        modality = "image"
    elif "roc" in lst:
        modality = "roc"
    elif "e2e-tgt" in lst:
        modality = "e2e-tgt"
    elif "simple-wiki" in lst:
        modality = "simple-wiki"
    elif "book" in lst:
        modality = "book"
    elif "yelp" in lst:
        modality = "yelp"
    elif "commonGen" in lst:
        modality = "commonGen"
    elif "e2e" in lst:
        modality = "e2e"

    if "synth32" in lst:
        kk = 32
    elif "synth128" in lst:
        kk = 128

    try:
        diffusion_steps = int(lst.split("_")[7 - num])
    except:
        diffusion_steps = 4000
    try:
        noise_schedule = lst.split("_")[8 - num]
        assert noise_schedule in ["cosine", "linear"]
    except:
        noise_schedule = "cosine"
    try:
        dim = 64
    except:
        dim = lst.split("_")[4 - num]
    try:
        num_channels = int(lst.split("_")[-1].split("h")[1])
    except:
        num_channels = 128

    out_dir = "../results/generation_outputs/" + os.path.join(lst.split('/')[-1],constrained)
    folder = os.path.exists(out_dir)
    if not folder:
        os.makedirs(out_dir)
    if constrained is None:
        COMMAND = (
            f"python scripts/{mode}_sample.py "
            f"--model_path {tgt} --batch_size {bz} --num_samples {num_samples} --top_p {top_p} "
            f"--out_dir {out_dir} --multistep {multistep}"
        )
        print(COMMAND)
    # os.system(COMMAND)
    else:
        COMMAND = (
            f"python scripts/{mode}_sample.py "
            f"--model_path {tgt} --batch_size {bz} --num_samples {num_samples} --top_p {top_p} "
            f"--out_dir {out_dir} --multistep {multistep} --constrained {constrained}"
        )
        print(COMMAND)

    # shape_str = "x".join([str(x) for x in arr.shape])
    model_base_name = (
        os.path.basename(os.path.split(tgt)[0]) + f".{os.path.split(tgt)[1]}"
    )
    if modality == "e2e-tgt" or modality == "e2e":
        out_path2 = os.path.join(out_dir, f"{model_base_name}.samples_{top_p}.json")
    else:
        out_path2 = os.path.join(out_dir, f"{model_base_name}.samples_{top_p}.txt")
    output_cands = glob.glob(out_path2)
    if len(output_cands) > 0:
        out_path2 = glob.glob(out_path2)[0]
    else:
        os.system(COMMAND)

print("\n".join(output_lst))