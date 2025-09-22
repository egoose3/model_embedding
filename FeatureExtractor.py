import torch
import torch.nn as nn
from torchvision import models


class FeatureExtractor(nn.Module):
    def __init__(self, model, returned_layer, output_dim=256, keep_original_dim=False):
        super(FeatureExtractor, self).__init__()
        self.keep_original_dim = keep_original_dim
        self.model = model
        self.returned_layer = returned_layer
        self.output_dim = output_dim
        # 检查模型类型
        self.device = next(self.model.parameters()).device

        if isinstance(model, models.ResNet):
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            dummy_out = self._get_layer_output(dummy_input)
            in_dim = dummy_out.shape[1]
            if keep_original_dim:
                self.fc_output = nn.Identity()
            else:
                self.fc_output = nn.Linear(in_dim, output_dim)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc_output = self.fc_output.to(self.device)
            self.avgpool = self.avgpool.to(self.device)

        elif isinstance(model, models.VisionTransformer):
            if hasattr(model, 'head'):
                model.head = nn.Identity()
            self.cls_token = model.class_token.to(self.device)
            self.pos_embedding = model.encoder.pos_embedding.to(self.device)
            in_dim = self.pos_embedding.shape[-1]
            if keep_original_dim:
                self.fc_output = nn.Identity()
            else:
                self.fc_output = nn.Linear(in_dim, output_dim)

            self.fc_output = self.fc_output.to(self.device)

    def _get_layer_output(self, x):
        """获取目标层的输出（辅助函数）"""
        if x.device != self.device:
            x = x.to(self.device)

        with torch.no_grad():
            if isinstance(self.model, models.ResNet):
                for name, module in self.model.named_children():
                    x = module(x)
                    if name == self.returned_layer:
                        return x
            elif isinstance(self.model, models.VisionTransformer):
                x = self.model.conv_proj(x).flatten(2).transpose(1, 2)
                x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
                x = x + self.pos_embedding[:, :x.size(1), :]
                for name, module in self.model.encoder.named_children():
                    x = module(x)
                    if name == self.returned_layer:
                        return x
        return x

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)
        if isinstance(self.model, models.ResNet):
            x = self._get_layer_output(x)
            x = self.avgpool(x).flatten(1)
        elif isinstance(self.model, models.VisionTransformer):
            x = self._get_layer_output(x)
            x = x[:, 0, :]  # 取 CLS token
        return self.fc_output(x)


#%%
# # 假设你加载了一个预训练的 ResNet 模型
# resnet50 = models.resnet50(pretrained=True)
#
# # 指定要提取的层（例如 'layer2' 或 'layer4'）
# returned_layer = 'layer2'
#
# # 创建特征提取器
# feature_extractor = FeatureExtractor(resnet50, returned_layer, output_dim=256)
#
# # 输入图像张量（例如 224x224 的图像）
# input_image = torch.randn(1, 3, 224, 224)
#
# # 获取中间层输出
# output = feature_extractor(input_image)
#
# # 查看输出的层
# print(output.shape)  # 输出 layer2 的特征形状
#
#%%
# vit_b_16 = models.vit_b_16(pretrained=True)
# returned_layer = "encoder_layer_12"
# feature_extractor = FeatureExtractor(vit_b_16, returned_layer, output_dim=256, keep_original_dim=True)
#
# # 测试输入图像
# input_image = torch.randn(1, 3, 224, 224)
#
# # 提取特征
# output = feature_extractor(input_image)
#
# print("ViT 提取的特征形状：", output.shape)
