import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from train_compare import DISPLAY_NAME_MAP, build_model


def find_target_layer(model_name: str, model: torch.nn.Module):
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "densenet121":
        return model.features.norm5
    if model_name == "efficientnet_b0":
        return model.features[-1]
    if model_name == "cnn":
        return model.features[-2]
    raise ValueError(f"No target layer rule for model: {model_name}")


def load_image(image_path: Path, img_size: int):
    tfm = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    tensor = tfm(image).unsqueeze(0)
    return image, tensor


def tensor_to_heatmap(cam_tensor):
    cam = cam_tensor.squeeze().detach().cpu().numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    return cam


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM for a trained chest X-ray model.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--output", type=str, default="gradcam_result.png")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = ckpt["model_name"]
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 224)

    model = build_model(model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    target_layer = find_target_layer(model_name, model)
    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output

    def backward_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    image, tensor = load_image(Path(args.image), img_size)
    output = model(tensor)
    pred_idx = int(output.argmax(dim=1).item())

    model.zero_grad()
    output[0, pred_idx].backward()

    grads = gradients["value"]
    acts = activations["value"]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.interpolate(cam, size=(img_size, img_size), mode="bilinear", align_corners=False)
    heatmap = tensor_to_heatmap(cam)

    fh.remove()
    bh.remove()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image.resize((img_size, img_size)))
    ax.imshow(heatmap, cmap="jet", alpha=0.4)
    pred_name = DISPLAY_NAME_MAP.get(class_names[pred_idx], class_names[pred_idx])
    ax.set_title(f"Predicted: {pred_name}")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved Grad-CAM to {args.output}")


if __name__ == "__main__":
    main()
