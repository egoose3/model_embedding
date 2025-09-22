import argparse
import os
import sys
import extract


def main(argv):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Model Embedding")
    parser.add_argument("--input", type=str, help="input path")
    parser.add_argument("--output_dir",
                        type=str,
                        default=current_dir,  # 默认值为当前脚本所在目录
                        help="Output directory. Default: current script directory")
    parser.add_argument("--model", type=str, help="choose mmodel to extract features",
                        choices=["resNet18", "resnet34", "resnet50", "resnet101", "resnet152", "vit_b_16", "vit_b_32",
                                 "vit_l_16", "vit_l_32", "vit_h_14"]
                        , default="resnet50")
    parser.add_argument("--model_path", type=str, default=None, help="model_path")
    parser.add_argument("--layer_name", type=str, help="layer name", default=None)
    parser.add_argument("--model_output_dimension", type=int, help="Model embedded output dimension", default=256)
    parser.add_argument("--keep_original_dim", type=bool, help="keep original dimension", default=False)
    args = parser.parse_args(args=argv)
    mes = extract.extract_feature(model_name=args.model, input_dir=args.input, output_dir=args.output_dir,
                                  model_path=args.model_path,
                                  layer_name=args.layer_name, model_output_dimension=args.model_output_dimension,
                                  keep_original_dim=args.keep_original_dim)
    print(mes)


if __name__ == '__main__':
    main(sys.argv[1:])
