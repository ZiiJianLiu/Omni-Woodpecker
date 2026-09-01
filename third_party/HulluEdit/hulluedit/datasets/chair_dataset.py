"""
CHAIR Dataset Builder
Reference Nullu's sampling method to ensure evaluation consistency with other methods
"""
import os
import random
from typing import List, Dict, Any
from pycocotools.coco import COCO


class CHAIRDataset:
    """CHAIR Dataset Builder (reference Nullu/dataset/CHAIR.py)"""
    
    def __init__(
        self, 
        split: str = "val", 
        data_root: str = "data/coco",
        sampling: str = "random", 
        num_samples: int = 500,
        seed: int = 0
    ):
        """
        Args:
            split: "val" or "train"
            data_root: COCO data root directory
            sampling: "first" or "random"
            num_samples: Number of samples to sample
            seed: Random seed
        """
        self.split = split
        self.ann_path = os.path.join(data_root, f"annotations/instances_{split}2014.json")
        self.caption_path = os.path.join(data_root, f"annotations/captions_{split}2014.json")
        self.img_root = os.path.join(data_root, f"{split}2014")
        self.sampling = sampling
        self.num_samples = num_samples
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
    
    def get_data(self) -> List[Dict[str, Any]]:
        """
        Get data list
        
        Returns:
            List[Dict]: Each element contains image_id, image_path, question
        """
        coco = COCO(self.caption_path)
        
        img_ids = coco.getImgIds()
        
        # Sampling strategy
        if self.num_samples:
            if self.num_samples > len(img_ids):
                print(f"[WARN] num_samples {self.num_samples} exceeds total images ({len(img_ids)})")
                sampled_img_ids = img_ids
            elif self.sampling == "first":
                sampled_img_ids = img_ids[:self.num_samples]
            elif self.sampling == "random":
                sampled_img_ids = random.sample(img_ids, self.num_samples)
            else:
                raise ValueError(f"Unsupported sampling strategy: {self.sampling}")
        else:
            sampled_img_ids = img_ids
        
        # Build data list
        val_data = []
        for cur_img_id in sampled_img_ids:
            cur_img = coco.loadImgs(cur_img_id)[0]
            cur_img_path = cur_img["file_name"]
            val_data.append({
                "image_id": cur_img_id,
                "image_path": os.path.join(self.img_root, cur_img_path),
                "question": "Please describe this image in detail."
            })
        
        return val_data


def build_chair_dataset(
    split: str = "val",
    data_root: str = "data/coco",
    sampling: str = "random",
    num_samples: int = 500,
    seed: int = 0
) -> List[Dict[str, Any]]:
    """
    Build CHAIR dataset (convenience function)
    
    Returns:
        List[Dict]: Data list
    """
    dataset = CHAIRDataset(split, data_root, sampling, num_samples, seed)
    return dataset.get_data()
