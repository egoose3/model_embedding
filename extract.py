import os
import h5py
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import ImageFile

from tqdm import tqdm
from torchvision import models
from FeatureExtractor import FeatureExtractor

torch.manual_seed(42)


def extract_feature(model_name, input_dir, output_dir, layer_name, model_output_dimension, keep_original_dim,
                    model_path):
    transform = torchvision.transforms.Compose([

        # torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
        #                                          ratio=(3.0 / 4.0, 4.0 / 3.0)),
        # torchvision.transforms.RandomHorizontalFlip(),
        # torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4,
        #                                    saturation=0.4),
        # torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize([0.485, 0.456, 0.406],
        #                                  [0.229, 0.224, 0.225])
        torchvision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                 ratio=(3.0 / 4.0, 4.0 / 3.0)),
        torchvision.transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path is None:
        model = getattr(torchvision.models, model_name)(pretrained=True)
    else:
        # 先判断文件是否存在
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"权重文件不存在: {model_path}")
        # 加载模型结构
        model = getattr(torchvision.models, model_name)(pretrained=False)

        # 加载权重（严格匹配）
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # 如果是保存的 checkpoint，需要取出 model_state_dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint  # 如果本来就是纯 state_dict
        if isinstance(model, models.ResNet):
            ignore_prefix = "fc."
        elif "vit" in model_name:
            ignore_prefix = "heads.head"
        else:
            ignore_prefix = ""  # 默认不忽略

        if ignore_prefix:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith(ignore_prefix)}

            # 加载参数（允许跳过分类头）
        model.load_state_dict(state_dict, strict=False)

    if layer_name is None:
        if "resnet" in model_name.lower():
            layer_name = "layer4"
        elif "vit" in model_name.lower():
            layer_name = "encoder_layer_11"  # ViT 最后一层 encoder
        else:
            raise ValueError(f"无法为 {model_name} 自动设置默认 layer_name")
    else:
        # 检查用户给的 layer_name 是否在模型里
        valid_layer_names = [name for name, _ in model.named_modules()]
        if layer_name not in valid_layer_names:
            raise ValueError(
                f"指定的 layer_name='{layer_name}' 在模型 {model_name} 中不存在。\n"
                f"可用层包括（示例）：{[n for n in valid_layer_names if 'encoder_layer' in n or n.startswith('layer')][:20]} ..."
            )

    model.to(device)
    model.eval()
    # 使用分割后的数据集路径来创建DataLoader

    all_dataset = datasets.ImageFolder(root=input_dir, transform=transform)
    all_dataloader = DataLoader(all_dataset, batch_size=32, shuffle=True)
    labels = []
    for label_dir in os.listdir(input_dir):
        labels.append(label_dir)
    # 创建特征提取器
    feature_extractor = FeatureExtractor(model, layer_name, model_output_dimension, keep_original_dim).to(device)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # 注意：ResNet50的conv5_x（即layer4的最后一个block）没有单独命名，但可以通过layer4的输出访问
    initial_shapes = None
    for images, _ in all_dataloader:  # 这里不需要标签
        images = images.to(device)
        output_demo = feature_extractor(images)
        print(output_demo.shape)
        initial_shapes = output_demo.shape[1:]
        break

    os.makedirs(output_dir, exist_ok=True)
    # %%
    # 创建HDF5文件并初始化数据集
    with h5py.File(os.path.join(output_dir, "feature.hdf5"), 'w') as hdf5_file:
        for label in labels:
            group = hdf5_file.create_group(label)
            dataset_path = f"{label}/{layer_name}_features"
            hdf5_file.create_dataset(dataset_path, shape=(0, initial_shapes[0]),
                                     maxshape=(None, initial_shapes[0]), dtype='f4', compression="gzip")
            # 现在，写入特征
    with h5py.File(os.path.join(output_dir, "feature.hdf5"), 'a') as hdf5_file:
        for inputs, batch_labels in tqdm(all_dataloader, desc="Extracting and saving features"):
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = feature_extractor(inputs)
                print(outputs.shape)
                # 对每个样本
                for i, label in enumerate(batch_labels):
                    label_str = f'{labels[label]}'
                    dataset_path = f'{label_str}/{layer_name}_features'
                    dataset = hdf5_file[dataset_path]
                    # 直接写入，无需调整大小（因为设置了maxshape）
                    dataset.resize((dataset.shape[0] + 1,) + dataset.shape[1:])
                    dataset[-1, :] = outputs[i].cpu().numpy().astype('float32')
