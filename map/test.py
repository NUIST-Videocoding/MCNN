
import argparse
import builtins
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
from torch.utils.data import DataLoader
from tqdm import tqdm
import oid_mask_encoding as oid_mask_encoding
from src.dataset import OpenImageDataset_test
from src.model_test import FeatureCompressor as Model
#是否解码真实bin文件
DECODE = True


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    # ---------------------- Main Arguments ---------------------------------|

    parser.add_argument(
        "-q", "--quality", type=int, default=2, help="quality of the model (1, 2, 3, 4, 5, 6)"
    )
    parser.add_argument("-t", "--task", type=str, default="segmentation", help="detection or segmentation")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="",
        help="Dataset path (root), 'openImage/train' and 'openImage/val' directories should exist in the root dataset path",
    )
    # -----------------------------------------------------------------------|

    parser.add_argument(
        "-s", "--safe_load", type=int, default=0, help="1 for safe load"
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
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )

    parser.add_argument("--cuda", action="store_true", default=True, help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument(
        "--seed", type=float, default=2024, help="Set random seed for reproducibility"
    )
    parser.add_argument("--checkpoint_folder", type=str, default=r'', help="Path to a checkpoint")
    parser.add_argument(
        "-savedir", "--savedir", type=str, default=r"", help="save_dir"
    )
    parser.add_argument(
        "-savedir_bin", "--savedir_bin", type=str, default=r"", help="save_dir"
    )
    parser.add_argument('-input_folder', type=str,
                        default=r'', help='包含输入文件的文件夹路径')
    parser.add_argument('-output_oi_folder', type=str,
                        default=r'', help='生成输出文件的文件夹路径')
    parser.add_argument('-output_metrics_folder', type=str,
                        default=r'', help='Output file with csv metrics.')


    parser.add_argument('-selected_classes', type=str,
                        default=r'', help='包含指定类别的文件路径')
    parser.add_argument('-input_annotations_boxes', type=str,
                        default=r'', help='File with groundtruth boxes annotations.')
    parser.add_argument('-input_annotations_labels', type=str,
                        default=r'', help='File with groundtruth labels annotations.')
    parser.add_argument('-input_class_labelmap', type=str,
                        default=r'', help='Open Images Challenge labelmap.')
    parser.add_argument('-input_annotations_segm', type=str,
                        default=r'')

    args = parser.parse_args(argv)
    return args


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


class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, task, weights=None):
        super().__init__()
        self.lmbda = {
            1: 0.0125,
            2: 0.025,
            3: 0.125,
            4: 0.25,
            5: 0.375,
            6: 0.5,
        }

        self.mse = nn.MSELoss()
        self.weights = self.get_weights(weights)
        self.cos_sim = nn.CosineSimilarity(dim=-1)
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
        out["mse_loss"] = sum(
            [
                self.mse(recon_and_gt[0], recon_and_gt[1]) * self.weights[i]
                for i, recon_and_gt in enumerate(zip(output["features"], target))
            ]
        )
        out["cos_sim_loss"] = sum(
            [
                self.cos_sim(recon_and_gt[0].reshape([recon_and_gt[0].shape[0], recon_and_gt[0].shape[1], -1]),
                             recon_and_gt[1].reshape([recon_and_gt[0].shape[0], recon_and_gt[0].shape[1], -1])) *
                self.weights[i]
                for i, recon_and_gt in enumerate(zip(output["features"], target))
            ]
        ).mean()

        if DECODE:
            return out
        else:
            num_pixels = N * H * W

            out["bpp_loss"] = sum(
                (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
                for likelihoods in output["likelihoods"].values()
            )
            out["loss"] = self.lmbda[q] * (0.4 * (1 - out["cos_sim_loss"]) + out["mse_loss"]) + out["bpp_loss"]
            return out



def test(
    global_step, test_dataloader, compressor, task_model, criterion, args, output_fname, file_path,
):
    compressor.eval()
    device = next(compressor.parameters()).device
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    cos_sim_loss = AverageMeter()
    aux_loss = AverageMeter()
    psnr = AverageMeter()

    of = open(output_fname, 'w')
    of.write('ImageID,LabelName,Score,XMin,XMax,YMin,YMax,ImageWidth,ImageHeight,Mask\n')
    with torch.no_grad():
        for i, input in tqdm(enumerate(test_dataloader)):
            x = input[0]

            # name = os.path.split(input[1][0])[-1]
            name = input[1]
            # x = x.to(device)
            # Feature extraction
            inputs = [
                {"image": x['image'][0].to(device), "height": x['height'].item(), "width": x['width'].item()}
            ]

            processed_x = task_model.preprocess_image(inputs)
            features = task_model.backbone(processed_x.tensor)
            features = [features[f"p{i}"] for i in range(2, 7)]
            out_net = compressor(features, DECODE, rf'{args.savedir_bin}/{global_step}/{os.path.split(input[1][0])[-1]}.bin')
            out_features = out_net['features']
            reconstruct_feature = {}
            for i in range(0, 5):
                reconstruct_feature[f"p{i+2}"] = out_features[i]
            batched_inputs = inputs
            images = task_model.preprocess_image(batched_inputs)

            # x = torch.as_tensor([features]).cuda().float()
            if task_model.proposal_generator is not None:
                proposals, _ = task_model.proposal_generator(images, reconstruct_feature, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(task_model.device) for x in batched_inputs]

            results, _ = task_model.roi_heads(images, reconstruct_feature, proposals, None)


            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            outputs = task_model._postprocess(results, batched_inputs, images.image_sizes)[0]


#============
            coco_classes_fname = 'E:\whh\W-MSFC\map\map_util\coco_classes.txt'
            with open(coco_classes_fname, 'r') as f:
                coco_classes = f.read().splitlines()
            stemId = os.path.splitext(os.path.basename(name[0]))[0]
            # stemId = name[0]
            classes = outputs['instances'].pred_classes.to('cpu').numpy()
            scores = outputs['instances'].scores.to('cpu').numpy()
            bboxes = outputs['instances'].pred_boxes.tensor.to('cpu').numpy()
            H, W = outputs['instances'].image_size
            # convert bboxes to 0-1
            # detectron: x1, y1, x2, y2 in pixels
            bboxes = bboxes / [W, H, W, H]
            # OpenImage output x1, x2, y1, y2 in percentage
            bboxes = bboxes[:, [0, 2, 1, 3]]

            masks = outputs['instances'].pred_masks.to('cpu').numpy()

            for ii in range(len(classes)):
                coco_cnt_id = classes[ii]
                class_name = coco_classes[coco_cnt_id]

                rslt = [stemId, class_name, scores[ii]] + \
                       bboxes[ii].tolist()

                assert (masks[ii].shape[1] == W) and (masks[ii].shape[0] == H)
                rslt += [masks[ii].shape[1], masks[ii].shape[0],
                         oid_mask_encoding.encode_binary_mask(masks[ii]).decode('ascii')]

                o_line = ','.join(builtins.map(str, rslt))
                of.write(o_line + '\n')

            out_criterion = criterion(out_net, features, [1, _, H, W], args.quality)
            if DECODE:
                mse_loss.update(out_criterion["mse_loss"])
                aux_loss.update(compressor.aux_loss())
                psnr.update(
                    10 * (torch.log(1 * 1 / out_criterion["mse_loss"]) / math.log(10))
                )
                cos_sim_loss.update(out_criterion["cos_sim_loss"])
            else:
                mse_loss.update(out_criterion["mse_loss"])
                aux_loss.update(compressor.aux_loss())
                bpp_loss.update(out_criterion["bpp_loss"])
                loss.update(out_criterion["loss"])
                psnr.update(
                    10 * (torch.log(1 * 1 / out_criterion["mse_loss"]) / math.log(10))
                )
                cos_sim_loss.update(out_criterion["cos_sim_loss"])

        of.close()
    print(
        f"Test global_step {global_step}: Average losses:"
        f"\tTest Loss: {loss.avg:.3f} |"
        f"\tTest COSSIM loss: {cos_sim_loss.avg:.4f} |"
        f"\tTest MSE loss: {mse_loss.avg:.3f} |"
        f"\tTest PSNR: {psnr.avg:.3f} |"
        f"\tTest Bpp loss: {bpp_loss.avg:.4f} |"
        f"\tTest Aux loss: {aux_loss.avg:.2f}\n"
    )

    save_to_file(global_step, loss, mse_loss,cos_sim_loss, psnr, bpp_loss, aux_loss, file_path)

    return loss.avg





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


def build_test_dataset(args):

    # Detectron transform (relies on cv2 instead of Pillow)

    test_transforms = [
            T.ResizeShortestEdge([800, 800], 1333),
        ]

    if args.task == "detection":
        test_dataset = OpenImageDataset_test(args.dataset, transform=test_transforms, split="det_val/data")
    else:
        test_dataset = OpenImageDataset_test(args.dataset, transform=test_transforms, split="seg_val/data")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return test_dataloader

def get_quality(name):
    if '1' in name:
        return 1
    elif '2' in name:
        return 2
    if '3' in name:
        return 3
    if '4' in name:
        return 4
    if '5' in name:
        return 5
    if '6' in name:
        return 6
    return 0


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    print("Dataset load")
    test_dataloader = build_test_dataset(args)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print("Model load")


    task_model = build_detectron(args.task)
    task_model.cuda().eval()
    criterion = RateDistortionLoss(task=args.task, weights=args.weights)

    global_step = 0

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    if not os.path.exists(args.savedir_bin):
        os.makedirs(args.savedir_bin)

    if args.checkpoint_folder:  # load from previous checkpoint
        print("Loading folder", args.checkpoint_folder)
        checkpoint_list = os.listdir(args.checkpoint_folder)

        for checkpoint_name in checkpoint_list:
            print("Loading checkpoint", checkpoint_name)
            compressor = Model(M=192)
            compressor = compressor.cuda()
            checkpoint_path = os.path.join(args.checkpoint_folder,checkpoint_name)
            simple_name_checkpoint = os.path.split(checkpoint_name)[-1][:-4]
            checkpoint = torch.load(checkpoint_path, map_location=device)
            output_fname = os.path.join(args.savedir,f"{simple_name_checkpoint}.txt")
            file_path = os.path.join(args.savedir,f"{simple_name_checkpoint}_bpp.txt")
            if os.path.exists(output_fname):
                print(f"Output {output_fname} file already exists! Skipping...")
                continue
            if args.safe_load:
                safe_load_state_dict(compressor, checkpoint["state_dict"])
            else:
                compressor.load_state_dict(checkpoint["state_dict"])
                global_step = checkpoint["global_step"]
                savedir_bin = os.path.join(args.savedir_bin,str(global_step))
                if not os.path.exists(savedir_bin):
                    os.makedirs(savedir_bin)
            compressor.eval()
            task_model.eval()
            test(
                global_step,
                test_dataloader,
                compressor,
                task_model,
                criterion,
                args,
                output_fname,
                file_path,
            )
            print(f"{simple_name_checkpoint} done!")
    """
  主函数。

  参数:
  input_folder (str): 输入文件夹路径。
  output_folder (str): 输出文件夹路径。
  selected_classes (str): 指定路径的某一个文件。
  """
    if not os.path.exists(args.output_oi_folder):
        os.makedirs(args.output_oi_folder)
    if not os.path.exists(args.output_metrics_folder):
        os.makedirs(args.output_metrics_folder)


    for filename in os.listdir(args.output_oi_folder):
    # for filename in os.listdir(args.input_folder):
        # if filename.endswith('bpp.txt'):
        #     continue
        # if filename.endswith('.bin'):
        #     continue
        # input_file = os.path.join(args.input_folder, filename)
        output_oi_file = os.path.join(args.output_oi_folder, filename)
        # if os.path.exists(output_oi_file):
        #     print(f"Output oi file already exists! Skipping...")
        #     continue
        # run_coco2oid(input_file, output_oi_file, args.selected_classes)

        output_result_file = os.path.join(args.output_metrics_folder, filename)
        # if os.path.exists(output_result_file):
        #     print(f"Output result file already exists! Skipping...")
        #     continue
        run_oid_challenge_evaluation(args.input_annotations_boxes, args.input_annotations_labels,
                                     args.input_class_labelmap, args.input_annotations_segm,output_oi_file, output_result_file)


def save_to_file(global_step, loss, mse_loss,cos_sim_loss, psnr, bpp_loss, aux_loss, file_path):
    with open(file_path, 'a') as file:
        file.write(
            f"Test global_step {global_step}: Average losses:"
            f"\tTest Loss: {loss.avg:.3f} |"
            f"\tTest MSE loss: {mse_loss.avg:.3f} |"
            f"\tTest COSSIM loss: {cos_sim_loss.avg:.4f} |"
            f"\tTest PSNR: {psnr.avg:.3f} |"
            f"\tTest Bpp loss: {bpp_loss.avg:.4f} |"
            f"\tTest Aux loss: {aux_loss.avg:.2f}\n"
        )


def safe_load_state_dict(model, pretrained_dict):
    new_model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    model.load_state_dict(new_model_dict)

import os
import subprocess
import argparse

from future.moves import sys



def run_coco2oid(coco_output_file, oid_output_file, selected_classes):
    """
  运行coco2oid.py程序。

  参数:
  coco_output_file (str): 输入文件路径。
  oid_output_file (str): 输出文件路径。
  selected_classes (str): 指定路径的某一个文件。
  """

    command = f"python E:\whh\VCM\W-MSFC\map\cvt_detectron_coco_oid.py --coco_output_file {coco_output_file} --oid_output_file {oid_output_file} --selected_classes {selected_classes}"
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print("Command executed successfully.")
        print("Output:")
        print(result.stdout)
    else:
        print("Command failed with error:")
        print(result.stderr)

def run_oid_challenge_evaluation(input_annotations_boxes, input_annotations_labels,input_class_labelmap,input_annotations_segm,input_predictions,output_metrics ):
    command = f"python E:\whh\VCM\W-MSFC\map\oid_challenge_evaluation.py \
            --input_annotations_boxes   {input_annotations_boxes} \
            --input_annotations_labels  {input_annotations_labels} \
            --input_class_labelmap      {input_class_labelmap} \
            --input_annotations_segm    {input_annotations_segm}\
            --input_predictions         {input_predictions} \
            --output_metrics            {output_metrics}"
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        print("Command executed successfully.")
        print("Output:")
        print(result.stdout)
    else:
        print("Command failed with error:")
        print(result.stderr)



if __name__ == "__main__":
    torch.cuda.set_device(0)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    main(sys.argv[1:])
