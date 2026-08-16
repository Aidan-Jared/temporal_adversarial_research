import argparse

from utils.best_args import best_args

parser = argparse.ArgumentParser()
parser.add_argument("--severity", type=int, default=1)
parser.add_argument("--poison_task", type=int, default=0)
parser.add_argument("--pcp", type=float, default=0.5)
parser.add_argument("--pp", type=float, default=0.5)
parser.add_argument(
    "--corruptions",
    nargs="+",
    default=["gaussian_blur"],
    type=str,
    help="The list of corruptions to apply to the poisoned samples, if not provided will apply no corruptions.",
    # choices=corruption_dict.keys(),
)

if __name__ == "__main__":
    args = vars(parser.parse_args())

    datasets = [
        "perm-mnist",
        "rot-mnist",
        "seq-mnist",
        "seq-cifar100",
        "cifar100-c",
        "seq-cifar10",
        "cifar10-c",
    ]
    models = ["gem", "agem", "der", "derpp", "xder"]  # "ewc_on",
    data_backbone = {
        "perm-mnist": ["mnistmlp", "mnistmlp-pnn"],
        "rot-mnist": ["mnistmlp", "mnistmlp-pnn"],
        "seq-mnist": ["mnistmlp", "mnistmlp-pnn"],
        "seq-cifar100": ["reduced-resnet18", "resnet18-pnn"],
        "cifar100-c": ["reduced-resnet18", "resnet18-pnn"],
        "seq-cifar10": ["reduced-resnet18", "resnet18-pnn"],
        "cifar10-c": ["reduced-resnet18", "resnet18-pnn"],
    }
    with open("args.txt", "a") as f:
        for data in datasets:
            params = best_args[data]
            for model in models:
                if model in params:
                    param = params[model]
                else:
                    continue
                for backbone in data_backbone[data]:
                    for buffer in param:
                        output = (
                            "--seed 42 "
                            + " --runs 5 "
                            + " --model_config base "
                            + " --dataset "
                            + data
                            + " --model "
                            + model.replace("_", "-")
                            + " --backbone "
                            + backbone
                            + " --severity "
                            + str(args["severity"])
                            + " --poison_task "
                            + str(args["poison_task"])
                            + " --pcp "
                            + str(args["pcp"])
                            + " --pp "
                            + str(args["pp"])
                            + " --corruptions "
                            + " ".join(args["corruptions"])
                        )
                        if buffer > 0:
                            output = output + f" --buffer_size {buffer} "
                        if "mnistmlp-pnn" in backbone:
                            output += f" --input_size {28 * 28} "
                        for arg, value in param[buffer].items():
                            output += f" --{arg} {value} "
                        # print(output)
                        f.write(output + "\n")
