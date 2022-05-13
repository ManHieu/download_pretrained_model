from huggingface_hub import snapshot_download
import os
import re


def download_to_local(repo_id: str, save_dir: str):
    model_dir = snapshot_download(repo_id=repo_id, ignore_regex=['*.msgpack', '*.ot', '*.h5'], cache_dir=save_dir, revision='main')
    new_model_dir = re.sub(f'{repo_id}.*', f'{repo_id}', model_dir)
    os.rename(model_dir, new_model_dir)
    return new_model_dir


if __name__ == '__main__':
    pretrains = ['roberta-base', 'bert-base-multilingual-uncased', 'roberta-large', 'xlm-roberta-base']
    save_dir = '/hf_hub'
    for model in pretrains:
        new_model_dir = download_to_local(model, save_dir)
        print(f"Loaded and saved {model} in {new_model_dir}")
