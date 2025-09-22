#%%
import torchvision
from torchvision import models

x = dir(models)
vit_b_16 = models.vit_b_16(pretrained=True)
print(vit_b_16)
#%%
y = vit_b_16.named_children()
z = models.resnet50(pretrained=True)

for name, child in y:
    if name == "encoder":
        for name_layer, layer in enumerate(child.layers):
            print()
    print(name)
#%%
from torch.hub import get_dir

# 查看PyTorch下载的默认目录
cache_dir = get_dir()
print(cache_dir)

#%%
import torch
import torchvision
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'resnet50'
model_path = r'E:\workspace\Neural networks\pth\best_mode_resnetl.pth'
model = getattr(torchvision.models, model_name)(pretrained=False)

checkpoint = torch.load(model_path, map_location=device)

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

