import argparse
import math
import os
import random
import sys

import detectron2.data.transforms as T
import torch
import torch.nn as nn
import torch.optim as optim

# detectron2
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from src.dataset import OpenImageDataset
from src.model import FeatureCompressor as Model
from src.model import FeatureCompressor as Model_No_AR


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    # ---------------------- Main Arguments ---------------------------------|
    parser.add_argument(
        "-q", "--quality", type=int, default=4, help="quality of the model (1, 2, 3, 4)"
    )
    parser.add_argument("-t", "--task", default='detection', type=str, help="segmentation or detection")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "-ps", "--patch-size", type=int, default=0, help="0 for no crop"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=f"openimage",
        help="Dataset path (root), 'openImage/train' and 'openImage/val' directories should exist in the root dataset path",
    )
    # -----------------------------------------------------------------------|

    parser.add_argument(
        "-s", "--safe_load", type=int, default=0, help="1 for safe load"
    )
    parser.add_argument(
        "-p", "--patience", type=int, default=200000, help="patience of scheduler"
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default=None,
        help="Ascending or Descending (asc or desc)",
    )

    parser.add_argument(
        "-o", "--only_model", type=int, default=0, help="Load only model weights"
    )
    parser.add_argument(
        "-savedir", "--savedir", type=str, default=rf"", help="save_dir"
    )
    parser.add_argument("-logdir", "--logdir", type=str, default=rf"", help="log_dir")
    parser.add_argument(
        "-total_step",
        "--total_step",
        default=400000,
        type=int,
        help="total_step (default: %(default)s)",
    )
    parser.add_argument(
        "-test_step",
        "--test_step",
        default=5000,
        type=int,
        help="test_step (default: %(default)s)",
    )
    parser.add_argument(
        "-acc_step",
        "--acc_step",
        default=1,
        type=int,
        help="accumulation_step (default: %(default)s)",
    )
    parser.add_argument(
        "-save_step",
        "--save_step",
        default=50000,
        type=int,
        help="save_step (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-5,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )

    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=2024, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", default=rf"", type=str, help="Path to a checkpoint")

    args = parser.parse_args(argv)
    return args


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, task, weights=None):
        super().__init__()
        self.lmbda = {
            1: 0.025,
            2: 0.125,
            3: 0.25,
            4: 0.5,
        }

        self.mse = nn.MSELoss()
        # self.ssim = ssim
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        # self.cos_sim = nn.CosineEmbeddingLoss()
        self.weights = self.get_weights(weights)

    def get_weights(self, weights):
        assert weights in ["asc, desc", None]

        if weights == "asc":
            return [0.0025, 0.01, 0.04, 0.16, 0.64]
        elif weights == "desc":
            return [0.64, 0.16, 0.04, 0.01, 0.0025]
        else:
            return [0.2, 0.2, 0.2, 0.2, 0.2]

    def forward(self, output, target, shape, q):
        N, _, H, W = shape
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )

        out["mse_loss"] = sum(
            [
                self.mse(recon_and_gt[0], recon_and_gt[1]) * self.weights[i]
                for i, recon_and_gt in enumerate(zip(output["features"], target))
            ]
        )

        out["cos_sim_loss"] = sum(
            [
                self.cos_sim(recon_and_gt[0].reshape([recon_and_gt[0].shape[0],recon_and_gt[0].shape[1],-1]),
                             recon_and_gt[1].reshape([recon_and_gt[0].shape[0],recon_and_gt[0].shape[1],-1])) * self.weights[i]
            for i, recon_and_gt in enumerate(zip(output["features"], target))
            ]
        ).mean()
        out["loss"] = self.lmbda[q] * (0.4*(1-out["cos_sim_loss"])+out["mse_loss"]) + out["bpp_loss"]
        return out


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def save_feature(title, feature):
    import matplotlib.pyplot as plt

    feature = feature[0]
    feature = feature[5]
    plt.imshow(feature.detach().cpu().numpy(), cmap=plt.cm.jet)

    shrink_scale = 1.0
    aspect = feature.shape[0] / float(feature.shape[1])
    if aspect < 1.0:
        shrink_scale = aspect
    plt.colorbar(shrink=shrink_scale)
    plt.clim(-8, 8)
    plt.tight_layout()
    plt.title(title)
    plt.show()
    plt.savefig(f"./image_result/{title}.png")
    plt.close()


def test(
    global_step, test_dataloader, compressor, task_model, criterion, logger, lr, args
):
    compressor.eval()
    device = next(compressor.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    cos_sim_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()

    with torch.no_grad():
        for i, x in tqdm(enumerate(test_dataloader)):
            x = x.to(device)
            # Feature extraction
            inputs = [
                {"image": x_, "height": x_.size()[1], "width": x_.size()[2]} for x_ in x
            ]

            processed_x = task_model.preprocess_image(inputs)
            features = task_model.backbone(processed_x.tensor)
            features = [features[f"p{i}"] for i in range(2, 7)]
            out_net = compressor(features)
            out_criterion = criterion(out_net, features, x.shape, args.quality)

            aux_loss.update(compressor.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])
            cos_sim_loss.update(out_criterion["cos_sim_loss"])
            psnr.update(
                10 * (torch.log(1 * 1 / out_criterion["mse_loss"]) / math.log(10))
            )

    print(
        f"Test global_step {global_step}: Average losses:"
        f"\tTest Loss: {loss.avg:.3f} |"
        f"\tTest COSSIM loss: {cos_sim_loss.avg:.4f} |"
        f"\tTest MSE loss: {mse_loss.avg:.3f} |"
        f"\tTest PSNR: {psnr.avg:.3f} |"
        f"\tTest Bpp loss: {bpp_loss.avg:.4f} |"
        f"\tTest Aux loss: {aux_loss.avg:.2f}\n"
    )

    logger.add_scalar("Test Loss", loss.avg, global_step)
    logger.add_scalar("Test COSSIM loss", cos_sim_loss.avg, global_step)
    logger.add_scalar("Test MSE loss", mse_loss.avg, global_step)
    logger.add_scalar("Test PSNR", psnr.avg, global_step)
    logger.add_scalar("Test Bpp loss", bpp_loss.avg, global_step)
    logger.add_scalar("Test Aux loss", aux_loss.avg, global_step)
    logger.add_scalar("lr", lr, global_step)
    return loss.avg


def train(
    compressor,
    task_model,
    criterion,
    train_dataloader,
    test_dataloader,
    optimizer,
    aux_optimizer,
    lr_scheduler,
    global_step,
    args,
    logger,
):
    compressor.train()
    device = next(compressor.parameters()).device
    best_loss = float("inf")

    for loop in range(100000):  # infinite loop
        for i, x in enumerate(tqdm(train_dataloader)):
            global_step += 1
            x = x.to(device)

            # Feature extraction
            with torch.no_grad():
                inputs = [
                    {"image": x_, "height": x_.size()[1], "width": x_.size()[2]}
                    for x_ in x
                ]
                processed_x = task_model.preprocess_image(inputs)
                features = task_model.backbone(processed_x.tensor)
                features = [features[f"p{i}"] for i in range(2, 7)]

            optimizer.zero_grad()
            aux_optimizer.zero_grad()

            q = args.quality
            out_net = compressor(features)

            out_criterion = criterion(out_net, features, x.shape, q)

            out_criterion["loss"].backward()
            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    compressor.parameters(), args.clip_max_norm
                )
            optimizer.step()

            aux_loss = compressor.aux_loss()
            aux_loss.backward()
            aux_optimizer.step()
            psnr = 10 * (torch.log(1 * 1 / out_criterion["mse_loss"]) / math.log(10))
            # Training log
            if global_step % 50 == 0:
                tqdm.write(
                    f"Train step \t{global_step}: \t["
                    f"{i * len(x)}/{len(train_dataloader.dataset)}"
                    f" ({50. * i / len(train_dataloader):.0f}%)]"
                    f'\tLoss: {out_criterion["loss"].item():.4f} |'
                    f'\tCOSSIM loss: {out_criterion["cos_sim_loss"].item():.3f} |'
                    f'\tMSE loss: {out_criterion["mse_loss"].item():.3f} |'
                    f"\tPSNR: {psnr.item():.3f} |"
                    f'\tBpp loss: {out_criterion["bpp_loss"].item():.4f} |'
                    f"\tAux loss: {aux_loss.item():.2f}"
                )

                logger.add_scalar("Loss", out_criterion["loss"].item(), global_step)
                logger.add_scalar(
                    "COSSIM loss", out_criterion["cos_sim_loss"].item(), global_step
                )
                logger.add_scalar(
                    "MSE loss", out_criterion["mse_loss"].item(), global_step
                )
                logger.add_scalar("PSNR", psnr.item(), global_step)
                logger.add_scalar(
                    "Bpp loss", out_criterion["bpp_loss"].item(), global_step
                )
                logger.add_scalar("Aux loss", aux_loss.item(), global_step)

            # validation
            if global_step % args.test_step == 0:
                loss = test(
                    global_step,
                    test_dataloader,
                    compressor,
                    task_model,
                    criterion,
                    logger,
                    optimizer.param_groups[0]["lr"],
                    args,
                )
                compressor.train()

                lr_scheduler.step(loss)

                is_best = loss < best_loss
                if is_best:
                    print("!!!!!!!!!!!BEST!!!!!!!!!!!!!")
                    os.makedirs(args.savedir, exist_ok=True)
                    save_checkpoint(
                        {
                            "global_step": global_step,
                            "state_dict": compressor.state_dict(),
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                        },
                        is_best,  # 另外保存最佳model
                        filename=f"{args.savedir}/best_{global_step}_checkpoint.pth",
                    )
                best_loss = min(loss, best_loss)

                if global_step % args.save_step == 0:
                    os.makedirs(args.savedir, exist_ok=True)
                    save_checkpoint(
                        {
                            "global_step": global_step,
                            "state_dict": compressor.state_dict(),
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                        },
                        is_best,  # 另外保存最佳model
                        filename=f"{args.savedir}/{global_step}_checkpoint.pth",
                    )

                # Early stop // 如果learning rate小于5e-6
                if (
                    optimizer.param_groups[0]["lr"] <= 5e-6
                    or args.total_step == global_step
                ):
                    os.makedirs(args.savedir, exist_ok=True)
                    print(
                        f'Finished. \tcurrent lr:{optimizer.param_groups[0]["lr"]} \tglobal step:{global_step}'
                    )
                    save_checkpoint(
                        {
                            "global_step": global_step,
                            "state_dict": compressor.state_dict(),
                            "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                        },
                        is_best,
                        filename=f"{args.savedir}/{global_step}_checkpoint.pth",
                    )
                    exit(0)


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    torch.save(state, filename)


def build_detectron(task="segmentation"):
    assert task in ["segmentation", "detection"]
    cfg = get_cfg()

    if task == "segmentation":
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
            )
        )
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        )
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        return model

    elif task == "detection":
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
            )
        )
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        )
        model = build_model(cfg)
        DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
        return model
    else:
        raise NotImplementedError


def build_dataset(args):
    if args.patch_size == 0 and args.batch_size != 1:
        raise NotImplementedError("Please use 1 batch-size for no cropping dataset")

    # Detectron transform (relies on cv2 instead of Pillow)
    train_transforms = [T.RandomFlip(prob=0.5, horizontal=True, vertical=False)]
    if args.patch_size:
        #===whh changed=====
        train_transforms = train_transforms + [
            T.RandomCrop("absolute", (args.patch_size, args.patch_size)),
        ]
        test_dataset = OpenImageDataset(args.dataset, split="val/data")
    else:
        train_transforms = train_transforms + [
            T.ResizeShortestEdge(short_edge_length=800, max_size=1333),
        ]
        test_dataset = OpenImageDataset(args.dataset, transform=train_transforms, split="val/data")
    train_dataset = OpenImageDataset(
        args.dataset, transform=train_transforms, split="train/data"
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    print("Dataset load")
    train_dataloader, test_dataloader = build_dataset(args)
    logger = SummaryWriter(args.logdir)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(device)
    print("Model load")

    compressor = Model_No_AR()

    compressor = compressor.to(device)
    task_model = build_detectron(args.task)
    task_model.to(device).eval()



    global_step = 0

    optimizer, aux_optimizer = configure_optimizers(compressor, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=round(args.patience / args.test_step) - 1,
    )

    criterion = RateDistortionLoss(task=args.task, weights=args.weights)

    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if args.safe_load:
            safe_load_state_dict(compressor, checkpoint["state_dict"])
        else:
            compressor.load_state_dict(checkpoint["state_dict"])

        if not args.only_model:
            global_step = checkpoint["global_step"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

    # if torch.cuda.device_count() > 1 :
    #     compressor = CustomDataParallel(compressor)
    #     task_model = CustomDataParallel(task_model)

    train(
        compressor,
        task_model,
        criterion,
        train_dataloader,
        test_dataloader,
        optimizer,
        aux_optimizer,
        lr_scheduler,
        global_step,
        args,
        logger,
    )


def safe_load_state_dict(model, pretrained_dict):
    new_model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    model.load_state_dict(new_model_dict)


if __name__ == "__main__":
    torch.cuda.set_device(1)
    main(sys.argv[1:])
