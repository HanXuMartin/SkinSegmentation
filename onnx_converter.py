# from configs import unet_cfg as cfg
import importlib.util
import os

import onnx
import onnxsim
import torch


def import_module_by_path(module_path):
    spec = importlib.util.spec_from_file_location("module_name", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    cfg = import_module_by_path(r"configs\fcn_cfg.py")
    input_size = cfg.input_size
    mymodel = cfg.model
    # checkpoint = r"workdir\Unet-00\Unet-epoch_9.pth"
    # mymodel.load_state_dict(torch.load(checkpoint))
    mymodel.eval()
    model_name = mymodel.__class__.__name__
    dummy_img = torch.randn(1, 3, input_size[1], input_size[0], requires_grad=True)
    onnx_path = f"./onnx_models/{model_name}.onnx"
    if not os.path.exists(os.path.dirname(onnx_path)):
        os.makedirs(os.path.dirname(onnx_path))
    
    input_names = ["input_data"]
    output_names = ['output_data']

    torch.onnx.export(
        mymodel,
        dummy_img,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
        opset_version=12
    )
    
    onnx_model = onnx.load(onnx_path)
    onnx_model_sim, check = onnxsim.simplify(onnx_path)
    onnx.save(onnx_model_sim, onnx_path)
    print(f"model save to: {onnx_path}")

if __name__ == "__main__":
    main()