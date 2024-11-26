import csv
import gc
import sys
import timeit
from torchvision import datasets, transforms, models
from tqdm import tqdm

sys.path.append("../")

import torch

def params():
    model = models.efficientnet_b0(pretrained=True)
    # model.load_state_dict(torch.load('/home/ubuntu/IndustrialDigitDatasetGenerator/best_model.pth'))
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total params:", total_params, "Trainable params:", trainable_params)

def prepare_image(batched=False):
    if batched:
        img = torch.randn(16, 3, 224, 224, dtype=torch.float16)
    else:
        img = torch.randn(1, 3, 224, 224, dtype=torch.float16)

    return img

def prepare_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
    elif model_name == "mnasnet1_0":
        model = models.mnasnet1_0(weights=models.MNASNet1_0_Weights.DEFAULT)
    elif model_name == "maxvit_t":
        model = models.maxvit_t(weights=models.MaxVit_T_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    model.to("cuda")
    model.to(torch.float16)
    model.eval()
    return model

@torch.no_grad()
def inference_speed(model_name, reps=1000):
    model = prepare_model(model_name)
    img = prepare_image()

    # first - warmup
    for i in tqdm(range(reps), desc="Warmup"):
        img = img.to("cpu")
        img = img.to("cuda")
        out = model(img)
        out = out.to("cpu")

    total_time = 0
    # next - real
    for i in tqdm(range(reps), desc="Timing inference"):
        img = img.to("cpu")
        t0 = timeit.default_timer()
        img = img.to("cuda")
        out = model(img)
        out = out.to("cpu")
        t1 = timeit.default_timer()
        total_time += t1 - t0

    # * 1000 to get ms
    ms = total_time * 1000 / reps
    print(f"Speed in ms ({model_name}):", ms)
    return ms


@torch.no_grad()
def throughput(model_name, reps=1000):
    model = prepare_model(model_name)
    img = prepare_image(batched=True)
    # first - warmup
    for i in tqdm(range(reps), desc="Warmup"):
        img = img.to("cpu")
        img = img.to("cuda")
        out = model(img)
        out = out.to("cpu")

    total_time = 0
    # next - real
    for i in tqdm(range(reps), desc="Throughput"):
        img = img.to("cpu")
        t0 = timeit.default_timer()
        img = img.to("cuda")
        out = model(img)
        out = out.to("cpu")
        t1 = timeit.default_timer()
        total_time += t1 - t0

    thru = 16 * reps / total_time
    print(f"Throughput ({model_name}):", thru)
    return thru


@torch.no_grad()
def memory(model_name, reps=1000):
    model = prepare_model(model_name)
    img = prepare_image()
    img = img.to("cuda")

    # first - warmup
    for i in tqdm(range(reps), desc="Warmup"):
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        out = model(img)

        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch._C._cuda_clearCublasWorkspaces()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    total_memory = 0
    # next - real
    for i in tqdm(range(reps), desc="Memory calc"):
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        out = model(img)

        total_memory += torch.cuda.max_memory_reserved()

        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # MB -> 10**6 bytes, then "reps" runs
    mbs = total_memory / (10**6) / reps
    print(f"Memory in MB ({model_name}):", mbs)
    return mbs


@torch.no_grad()
def flops(model_name, reps=1000):
    model = prepare_model(model_name)
    img = prepare_image()
    img = img.to("cuda")

    # first - warmup
    out = model(img)

    # real - don't need reps as the result is always same
    with torch.profiler.profile(with_flops=True) as prof:
        out = model(img)
    tflops = sum(x.flops for x in prof.key_averages()) / 1e9
    print(f"TFLOPS ({model_name}):", tflops)

    return tflops


def main():
    cycles = 2
    reps = 1000
    models_to_test = ["resnet50", "resnet18", "efficientnet_b0", "mnasnet1_0", "maxvit_t"]

    torch.backends.cudnn.deterministic = True

    for model_name in models_to_test:
        with open(f"perf_{model_name}.csv", "w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow(["time", "throughput", "memory", "tflops"])
            for cyc in range(cycles):
                ms = inference_speed(model_name, reps)
                thru = throughput(model_name, reps)
                mbs = memory(model_name, reps)
                tflops = flops(model_name, reps)

                if cyc == 0:
                    # skip first one, as the system is not warmed up and it's too fast
                    continue

                writer.writerow([ms, thru, mbs, tflops])
                print("-" * 42)
                print(f"Model: {model_name}")
                print("Speed [ms]:", ms)
                print("Throughput:", thru)
                print("Memory [MB]:", mbs)
                print("TFLOPS:", tflops)
                print("-" * 42)


if __name__ == "__main__":
    main()
    # params()
