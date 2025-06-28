import argparse
import torch

from ultralytics import YOLO


n_gpus = torch.cuda.device_count()
print(f"Detected {n_gpus} GPUs.")


def main(opt):
#    yaml = opt.cfg
#    model = YOLO('ultralytics/cfg/models/v8/yolov8-dww.yaml') 
    model = YOLO('yolov8s.yaml') 

    results = model.train(data='yolo_du_seeship.yaml', 
                        epochs=100, 
                        batch=64,
#                        device=[0,1],  # 指定GPU索引列表
#                        workers=0,
                        imgsz=320,
                        project='./my_train_logs',
                        name='exp',  # 或其他名称
                        exist_ok=False  # 如果目录已存在则报错（避免覆盖）
                        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)