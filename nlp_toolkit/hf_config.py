from pydantic import BaseSettings
import os

class HfSettings(BaseSettings):
    HF_HOME:str = "/Users/wanghuan/Projects/huggingface/hf_cache_dir"
    HF_DATASETS_CACHE:str = '/Users/wanghuan/Projects/huggingface/hf_cache_dir/datasets'
    TRANSFORMERS_CACHE:str = "/Users/wanghuan/Projects/huggingface/hf_cache_dir/models"
 
settings = HfSettings()

def set_huggingface_dir():
    for k,v in settings.dict().items():
        os.environ[k] = v
        
    print("huggingface home: {}".format(settings.HF_HOME))
    print("huggingface datasets cache: {}".format(settings.HF_DATASETS_CACHE))
    print("huggingface models cache: {}".format(settings.TRANSFORMERS_CACHE))
        

def set_proxy(port:int = 8001):
    port = str(port)
    os.environ["http_proxy"] = f"http://127.0.0.1:{port}"
    os.environ["https_proxy"] = f"http://127.0.0.1:{port}"
    os.environ['all_proxy'] = f"http://127.0.0.1:{port}"
    print("Using http proxy:", os.environ["http_proxy"])
    print("Turn on VPN with corresponding proxy.")