import argparse
import time
import sys
import cv2
import glob
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
import torch

sys.path.append(".")
from model.image_process.Real_ESRGAN.realesrgan.utils import RealESRGANer
from model.image_process.Real_ESRGAN.realesrgan.archs.srvgg_arch import SRVGGNetCompact
import model.image_process.dehazer as dehazer


realesr_parser = argparse.ArgumentParser()
realesr_parser.add_argument('-i', '--input', type=str, default='inputs', help='Input image or folder')
realesr_parser.add_argument(
    '-n',
    '--model_name',
    type=str,
    default='realesr-general-x4v3',
    help=('Model names: RealESRGAN_x4plus | RealESRNet_x4plus | RealESRGAN_x4plus_anime_6B | RealESRGAN_x2plus | '
          'realesr-animevideov3 | realesr-general-x4v3'))
realesr_parser.add_argument('-o', '--output', type=str, default='results2', help='Output folder')
# use dni to control the denoise strength
realesr_parser.add_argument(
    '-dni',
    '--denoise_strength',
    type=float,
    default=0.01,
    help=('Denoise strength. 0 for weak denoise (keep noise), 1 for strong denoise ability. '
          'Only used for the realesr-general-x4v3 model'))
realesr_parser.add_argument('-s', '--outscale', type=float, default=1, help='The final upsampling scale of the image')
realesr_parser.add_argument(
    '--model_path', type=str, default=None, help='[Option] Model path. Usually, you do not need to specify it')
realesr_parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
realesr_parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
realesr_parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
realesr_parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
realesr_parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
realesr_parser.add_argument(
    '--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
realesr_parser.add_argument(
    '--alpha_upsampler',
    type=str,
    default=None,
    help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
realesr_parser.add_argument(
    '--ext',
    type=str,
    default='auto',
    help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
realesr_parser.add_argument(
    '-g', '--gpu-id', type=int, default=0, help='gpu device to use (default=None) can be 0,1,2 for multi-gpu')

realesr_args = realesr_parser.parse_args()


class Denoiser:
    @staticmethod
    def process(image):
        realesr_args.model_name='realesr-general-x4v3'
        realesr_args.dni=0.99
        # determine models according to model names
        realesr_args.model_name = realesr_args.model_name.split('.')[0]
        if realesr_args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif realesr_args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 2
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]

        # determine model paths
        if realesr_args.model_path is not None:
            model_path = realesr_args.model_path
        else:
            model_path = os.path.join('weights', realesr_args.model_name + '.pth')
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    # model_path will be updated
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, "Real_ESRGAN", 'weights'), progress=True, file_name=None)

        device = None
        dni_weight = None
        if realesr_args.model_name == 'realesr-general-x4v3' and realesr_args.denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [realesr_args.denoise_strength, 1 - realesr_args.denoise_strength]

        gpu_id = realesr_args.gpu_id
        # initialize model
        if gpu_id:
            device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if isinstance(model_path, list):
            # dni
            assert len(model_path) == len(dni_weight), 'model_path and dni_weight should have the save length.'
            loadnet = RealESRGANer.dni(self=None, net_a=model_path[0], net_b=model_path[1], dni_weight=dni_weight)
        else:
            # if the model_path starts with https, it will first download models to the folder: weights
            if model_path.startswith('https://'):
                model_path = load_file_from_url(
                    url=model_path, model_dir=os.path.join(ROOT_DIR, "Real_ESRGAN", 'weights'), progress=True, file_name=None)
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))

        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=realesr_args.tile,
            tile_pad=realesr_args.tile_pad,
            pre_pad=realesr_args.pre_pad,
            half=not realesr_args.fp32,
            gpu_id=realesr_args.gpu_id,
            loadnet=loadnet,
            device=device)

        if realesr_args.face_enhance:  # Use GFPGAN for face enhancement
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=realesr_args.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)
        os.makedirs(realesr_args.output, exist_ok=True)

        if os.path.isfile(realesr_args.input):
            paths = [realesr_args.input]
        else:
            paths = sorted(glob.glob(os.path.join(realesr_args.input, '*')))

        # 按照最长边等比缩放,如果x长则按照1024/x的比例缩放,反之按照1024/y缩放
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if img.shape[0] >= 1024 or img.shape[1] >= 1024:

            if img.shape[0] > img.shape[1]:
                fx = fy = 1024 / img.shape[0]
            else:
                fx = fy = 1024 / img.shape[1]
            img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        if realesr_args.face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False,
                                                 paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=realesr_args.outscale)
            return output

class Inpainter:
    @staticmethod
    def process(image):
        realesr_args.model_name='RealESRGAN_x4plus'
        realesr_args.outscale=1
        # determine models according to model names
        realesr_args.model_name = realesr_args.model_name.split('.')[0]
        if realesr_args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif realesr_args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 2
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]

        # determine model paths
        # determine model paths
        if realesr_args.model_path is not None:
            model_path = realesr_args.model_path
        else:
            model_path = os.path.join('weights', realesr_args.model_name + '.pth')
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    # model_path will be updated
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, "Real_ESRGAN", 'weights'), progress=True, file_name=None)

        device = None
        dni_weight = None
        if realesr_args.model_name == 'realesr-general-x4v3' and realesr_args.denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [realesr_args.denoise_strength, 1 - realesr_args.denoise_strength]

        gpu_id = realesr_args.gpu_id
        # initialize model
        if gpu_id:
            device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if isinstance(model_path, list):
            # dni
            assert len(model_path) == len(dni_weight), 'model_path and dni_weight should have the save length.'
            loadnet = RealESRGANer.dni(self=None, net_a=model_path[0], net_b=model_path[1], dni_weight=dni_weight)
        else:
            # if the model_path starts with https, it will first download models to the folder: weights
            if model_path.startswith('https://'):
                model_path = load_file_from_url(
                    url=model_path, model_dir=os.path.join(ROOT_DIR, "Real_ESRGAN", 'weights'), progress=True, file_name=None)
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))

        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=realesr_args.tile,
            tile_pad=realesr_args.tile_pad,
            pre_pad=realesr_args.pre_pad,
            half=not realesr_args.fp32,
            gpu_id=realesr_args.gpu_id,
            loadnet=loadnet,
            device=device)

        if realesr_args.face_enhance:  # Use GFPGAN for face enhancement
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=realesr_args.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)
        os.makedirs(realesr_args.output, exist_ok=True)

        if os.path.isfile(realesr_args.input):
            paths = [realesr_args.input]
        else:
            paths = sorted(glob.glob(os.path.join(realesr_args.input, '*')))

        # 按照最长边等比缩放,如果x长则按照1024/x的比例缩放,反之按照1024/y缩放
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if img.shape[0] >= 1024 or img.shape[1] >= 1024:

            if img.shape[0] > img.shape[1]:
                fx = fy = 1024 / img.shape[0]
            else:
                fx = fy = 1024 / img.shape[1]
            img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        if realesr_args.face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False,
                                                 paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=realesr_args.outscale)
            return output

class SuperRestorater:
    @staticmethod
    def process(image):
        realesr_args.model_name='realesr-general-x4v3'
        realesr_args.outscale=2
        # determine models according to model names
        realesr_args.model_name = realesr_args.model_name.split('.')[0]
        if realesr_args.model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif realesr_args.model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 2
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]

        # determine model paths
        # determine model paths
        if realesr_args.model_path is not None:
            model_path = realesr_args.model_path
        else:
            model_path = os.path.join('weights', realesr_args.model_name + '.pth')
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    # model_path will be updated
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, "Real_ESRGAN", 'weights'), progress=True, file_name=None)

        device = None
        dni_weight = None
        if realesr_args.model_name == 'realesr-general-x4v3' and realesr_args.denoise_strength != 1:
            wdn_model_path = model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            model_path = [model_path, wdn_model_path]
            dni_weight = [realesr_args.denoise_strength, 1 - realesr_args.denoise_strength]

        gpu_id = realesr_args.gpu_id
        # initialize model
        if gpu_id:
            device = torch.device(
                f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu') if device is None else device
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if isinstance(model_path, list):
            # dni
            assert len(model_path) == len(dni_weight), 'model_path and dni_weight should have the save length.'
            loadnet = RealESRGANer.dni(self=None, net_a=model_path[0], net_b=model_path[1], dni_weight=dni_weight)
        else:
            # if the model_path starts with https, it will first download models to the folder: weights
            if model_path.startswith('https://'):
                model_path = load_file_from_url(
                    url=model_path, model_dir=os.path.join(ROOT_DIR, "Real_ESRGAN", 'weights'), progress=True, file_name=None)
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))

        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=dni_weight,
            model=model,
            tile=realesr_args.tile,
            tile_pad=realesr_args.tile_pad,
            pre_pad=realesr_args.pre_pad,
            half=not realesr_args.fp32,
            gpu_id=realesr_args.gpu_id,
            loadnet=loadnet,
            device=device)

        if realesr_args.face_enhance:  # Use GFPGAN for face enhancement
            from gfpgan import GFPGANer
            face_enhancer = GFPGANer(
                model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                upscale=realesr_args.outscale,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler)
        os.makedirs(realesr_args.output, exist_ok=True)

        if os.path.isfile(realesr_args.input):
            paths = [realesr_args.input]
        else:
            paths = sorted(glob.glob(os.path.join(realesr_args.input, '*')))

        # 按照最长边等比缩放,如果x长则按照1024/x的比例缩放,反之按照1024/y缩放
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
        if img.shape[0] >= 1024 or img.shape[1] >= 1024:

            if img.shape[0] > img.shape[1]:
                fx = fy = 1024 / img.shape[0]
            else:
                fx = fy = 1024 / img.shape[1]
            img = cv2.resize(img, (0, 0), fx=fx, fy=fy, interpolation=cv2.INTER_AREA)

        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        if realesr_args.face_enhance:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=realesr_args.outscale)
            return output

class Dehazer:
    @staticmethod
    def process(image):
        HazeImg = cv2.imread(image)  # read input image -- (**must be a color image**)
        output, haze_map = dehazer.remove_haze(HazeImg, showHazeTransmissionMap=False)  # Remove Haze
        return output

if __name__ == '__main__':
    img = './model/image_process/source/foggy-school-morning.jpg'
    img = SuperRestorater.process(img)
    cv2.imshow("res", img)
    cv2.waitKey()
