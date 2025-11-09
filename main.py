import argparse
import yaml
from easydict import EasyDict
import os
import glob

from utils.sse import sse_input_path_validated, sse_output_path_validated
from utils.yaml_rw import load_yaml, save_yaml
from ssd_detector.main import main as ssd_main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./input', help='input path')
    parser.add_argument('--output_path', type=str, default='./output', help='output path')
    
    parser.add_argument('--process', type=str, default='attack', help='[adv, attack, defend, train]')
    parser.add_argument('--model', type=str, default='ssd300', help='model name [ssd300]')
    parser.add_argument('--backbone', type=str, default='vgg16', help='backbone [vgg16]')
    parser.add_argument('--data', type=str, default='coco', help='data name [coco, voc]')
    parser.add_argument('--task', type=str, default='detect', help='task name [detect]')
    parser.add_argument('--class_number', type=int, default=80, help='number of class [80 for coco, 20 for voc]')
    
    parser.add_argument('--attack_method', type=str, default='fgsm', help='attack method [cw, deepfool, bim, fgsm, pgd]')
    parser.add_argument('--defend_method', type=str, default='scale', help='defend method [scale, comp, neural_cleanse, pgd, fgsm]')
    
    parser.add_argument('--cfg_path', type=str, default='./cfgs', help='cfg path')
    
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--device', type=int, default=0, help='which gpu for cuda')
    parser.add_argument('--workers', type=int, default=0, help='dataloader workers')
    
    parser.add_argument('--epsilon', type=float, default=8/255, help='epsilon for attack method')
    parser.add_argument('--step_size', type=float, default=2/255, help='step size for attack method')
    parser.add_argument('--max_iterations', type=int, default=50, help='max iterations for attack method')
    parser.add_argument('--random_start', type=bool, default=False, help='initial random start for attack method')
    parser.add_argument('--loss_function', type=str, default='CrossEntropy', help='loss function for attack method')
    parser.add_argument('--optimization_method', type=str, default='Adam', help='optimization for attack method')
    
    args = parser.parse_args()
    args_dict = vars(args)
    args_dict_environ = {}
    for key, value in args_dict.items():
        args_dict_environ[key] = type_switch(os.getenv(key.upper(), value), value)
    args_easydict = EasyDict(args_dict_environ)
    return args_easydict


def type_switch(environ_value, value):
    if isinstance(value, int):
        return int(environ_value)
    elif isinstance(value, float):
        return float(environ_value)
    elif isinstance(value, bool):
        return bool(environ_value)
    elif isinstance(value, str):
        return environ_value
    
def ssd_cfg(args):
    os.makedirs(args.cfg_path, exist_ok=True)
    
    cfg = EasyDict()
    cfg.task = args.task
    cfg.model = args.model
    cfg.backbone = args.backbone
    cfg.data = args.data
    cfg.save_dir = args.output_path
    cfg.project = args.model
    cfg.name = args.process
    cfg.batch = args.batch
    cfg.workers = args.workers
    cfg.device = f'cuda:{args.device}' if args.device >= 0 else 'cpu'
    cfg.num_classes = args.class_number
    cfg.verbose = True
    cfg.half = False
    cfg.split = 'val'
    
    if args.process == 'adv':
        cfg.mode = 'adv'
        cfg.batch = 1
        cfg.pretrained = glob.glob(os.path.join(f'{args.input_path}/model', '*'))[0]
        cfg.data_path = glob.glob(os.path.join(f'{args.input_path}/data', '*/'))[0]
    elif args.process == 'attack':
        cfg.mode = 'attack'
        cfg.batch = 1
        cfg.pretrained = glob.glob(os.path.join(f'{args.input_path}/model', '*'))[0]
        cfg.data_path = glob.glob(os.path.join(f'{args.input_path}/data', '*/'))[0]
    elif args.process == 'defend':
        cfg.mode = 'defend'
        cfg.batch = 1
        cfg.device = 'cpu'
        cfg.data_path = glob.glob(os.path.join(f'{args.input_path}/data', '*/'))[0]
    elif args.process == 'train':
        cfg.mode = 'train'
        cfg.epochs = args.epochs
        cfg.data_path = glob.glob(os.path.join(f'{args.input_path}/data', '*/'))[0]
    
    cfg_dict = dict(cfg)
    args.cfg_yaml = f'{args.cfg_path}/config.yaml'
    save_yaml(cfg_dict, args.cfg_yaml)
    
    return args, cfg

def main(args):
    args, cfg = ssd_cfg(args)
    ssd_main(args, cfg)
        
if __name__ == '__main__':
    args = parse_args()
    
    sse_input_path_validated(args)
    sse_output_path_validated(args)
    main(args)

