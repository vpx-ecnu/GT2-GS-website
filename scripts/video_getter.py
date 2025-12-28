from pathlib import Path
from dataclasses import dataclass
from dataclasses import field
import numpy as np
from PIL import Image
import imageio
import torch

from typing import Any, List, Dict, Callable

from video_processors import PROCESSORS

@dataclass
class VideoGetterConfig:

    fps: int = 30

    base_path: str = "/ckpt"
    save_path: str = "/workspace/assets/video"

    methods: list[str] = field(default_factory=lambda :["ABCGS", "GT2GS", "SGSST", "StyleGaussian", "arf", "refnpr", "snerf"])
    resize_scale: float = 4

    device: str = "cpu"
    dtype: torch.dtype = torch.float32

    processors: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "to_tensor"},
    ])

    rewrite: bool = False


class VideoGetter:

    def __init__(self, cfg = VideoGetterConfig()):

        self.cfg = cfg
        self.base_path = Path(cfg.base_path)
        self.save_path = Path(cfg.save_path)
        self.fps = cfg.fps

        self.processors = self._build_processors()

    def _build_processors(self) -> List[Callable]:
        processors = []
        for proc_cfg in self.cfg.processors:
            name = proc_cfg["name"]
            assert name in PROCESSORS, f"Unknown processor '{name}'. Available: {list(PROCESSORS.keys())}"

            if name == "temporal_sample":
                self.fps /= proc_cfg['step']
            
            bound_kwargs = {
                **self.cfg.__dict__,
                **proc_cfg,
            }
            processors.append(lambda frames, fn=PROCESSORS[name], k=bound_kwargs: fn(frames, **k))
        return processors


    def get_video(self, method, scene_type, scene_name, style_type, style_name):

        assert method in self.cfg.methods, f"{method} is not in pre-defined methods' list {self.cfg.methods}."
        path = self.base_path / method / scene_type / scene_name / style_type / style_name / "video.mp4"
        return self.get_video_by_path(path)
    
    
    def get_video_by_path(self, path: Path):

        path = Path(path)

        assert path.exists(), f"Path {path} does not exists."

        print(f"Reading Video From {path}...")
        frames = imageio.mimread(path, memtest=False)

        for i, proc in enumerate(self.processors):
            frames = proc(frames)
            print(f"  → Processor {i+1}: {self.cfg.processors[i]}")

        return frames
    
    def process_video(self, method, scene_type, scene_name, style_type, style_name):

        read_path = self.base_path / method / scene_type / scene_name / style_type / style_name / "video.mp4"
        save_path = self.save_path / method / scene_type / scene_name / style_type / style_name / "video.mp4"


        self.process_video_by_path(read_path, save_path)

    
    def process_video_by_path(self, read_path, save_path: Path):
        
        read_path = Path(read_path)
        save_path = Path(save_path)
        if save_path.exists() and not self.cfg.rewrite: 
            print(f"File {save_path} has already exists, skip processing.")
            return

        video = self.get_video_by_path(read_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        frames = [np.array(f) for f in video]
        print(f"Save Video (fps={self.fps}) in {save_path}")
        imageio.mimsave(save_path, frames, fps=self.fps)

    def process_image_by_path(self, read_path, save_path: Path):

        read_path = Path(read_path)
        save_path = Path(save_path)
        if save_path.exists() and not self.cfg.rewrite: 
            print(f"File {save_path} has already exists, skip processing.")
            return
        frames = [imageio.imread(read_path)]        
        for i, proc in enumerate(self.processors):
            frames = proc(frames)
            print(f"  → Processor {i+1}: {self.cfg.processors[i]}")
        image = np.array(frames[0])
        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Save Image in {save_path}")
        imageio.imsave(save_path, image)
        
def generate_html(scene_type, scene_name, style_type, style_name):

    all_path = Path("./assets/compare") / scene_type / scene_name / style_type / style_name / "all.jpg"
    assert all_path.exists(), f"File {all_path} does not exists."

    methods_dict = {
        "GT2GS": "Ours",
        "SGSST": "SGSST",
        "ABCGS": "ABCGS",
        "StyleGaussian": "StyleGaussian",
        "arf": "ARF",
        "refnpr": "RefNPR",
        "snerf": "SNeRF",
    }
    video_paths = []
    for k, v in methods_dict.items():
        video_path = Path("./assets/video") / k / scene_type / scene_name / style_type / style_name / "video.mp4"
        assert video_path.exists(), f"File {all_path} does not exists."
        video_paths.append((video_path, v))


    html_text = f"""
            <div class="item">
            <img src="{all_path}" style="width: 100%; height: auto;">
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
"""
    
    for (p, name) in video_paths:
        html_text += f"""
                <figure style="flex: 0 0 calc({100 / len(video_paths) - 0.3:.6f}%); margin: 0; text-align: center;">
                    <video autoplay loop muted playsinline controls style="width: 100%; height: auto; max-height: 180px; object-fit: contain; background: #000;">
                        <source src="{p}" type="video/mp4">
                    </video>
                    <figcaption style="display: block; margin-top: 6px; font-size: 13px; color: #333;">{name}</figcaption>
                </figure>
        """
    html_text += """
            </div>
            </div>
"""
    return html_text


if __name__ == "__main__":

    compare_list = [
        ["llff", "fern", "texture", "grid_0071.jpg"],
        ["llff", "fortress", "texture", "grid_0049.jpg"],
        ["llff", "horns", "texture", "banded_0154.jpg"],
        ["llff", "horns", "texture", "grid_0002.jpg"],
        ["llff", "orchids", "texture", "banded_0023.jpg"],
        ["tnt", "family", "texture", "chequered_0045.jpg"],
        ["tnt", "m60", "texture", "grid_0049.jpg"],
        ["tnt", "playground", "texture", "grooved_0065.jpg"],
        ["tnt", "truck", "texture", "grid_0002.jpg"],
        ["tnt", "truck", "texture", "grid_0066.jpg"],
        ["llff", "fern", "style", "0.jpg"],
        ["llff", "flower", "style", "92.jpg"],
        ["llff", "fortress", "style", "88.jpg"],
        ["llff", "trex", "style", "73.jpg"],
        ["llff", "trex", "style", "95.jpg"],
        ["tnt", "family", "style", "0.jpg"],
        ["tnt", "family", "style", "88.jpg"],
        ["tnt", "playground", "style", "46.jpg"],
        ["tnt", "train", "style", "42.jpg"],
        ["tnt", "truck", "style", "125.jpg"],
    ]

    cfg = VideoGetterConfig(
        base_path="/ckpt",
        processors=[
            {"name": "to_tensor"},

            # {"name": "temporal_sample", 
            #  "step": 2},
            
            {"name": "resize", 
             "scale": 0.25},
            
            # 最后转换为 tensor
            {"name": "to_pil"},
        ],
        device="cpu",
        dtype=torch.float32,
    )

    video_getter = VideoGetter(cfg)

    for method in cfg.methods:
        for args in compare_list:
            video_getter.process_video(method, *args)

    rotate_list = [
        ["/workspace/static/rotation/family_grid_0002.jpg.mp4", "/workspace/assets/rotate/family_grid_0002.mp4"],
        ["/workspace/static/rotation/fern_banded_0042.jpg.mp4", "/workspace/assets/rotate/fern_banded_0042.mp4"],
        ["/workspace/static/rotation/texture_rotate.mp4", "/workspace/assets/rotate/texture_rotate.mp4"],
        ["/workspace/static/rotation/trex_chequered_0045.jpg.mp4", "/workspace/assets/rotate/trex_chequered_0045.mp4"],
        ["/workspace/static/rotation/truck_banded_0154.jpg.mp4", "/workspace/assets/rotate/truck_banded_0154.mp4"],
    ]

    for (read_path, save_path) in rotate_list:
        video_getter.process_video_by_path(read_path, save_path)

    
    for (scene_type, scene_name, style_type, style_name) in compare_list:

        read_path = Path(cfg.base_path) / "GT2GS/concat/comparison/style" \
            / scene_type / scene_name / style_type / style_name / "train_frames/horizon_00000.png"
        save_path = Path(cfg.save_path).parent / "compare" / scene_type / scene_name / style_type / style_name / "all.jpg"
        video_getter.process_image_by_path(read_path, save_path)


    title_dict = {
        ("llff", "texture"): "Texture Transfer (LLFF Dataset)",
        ("tnt", "texture"): "Texture Transfer (T&T Dataset)",
        ("llff", "style"): "Style Transfer (LLFF Dataset)",
        ("tnt", "style"): "Style Transfer (T&T Dataset)",
    }

    html_text = ""
    for b in ("texture", "style"): 
        for a in ("llff", "tnt"):

            html_text += f"""
    <section class="hero is-small">
        <div class="hero-body">
        <div class="container">
            <h2 class="title is-3">{title_dict[(a, b)]}</h2>
            <div id="results-carousel" class="carousel results-carousel">
        """
            curr_list = list(filter(lambda x: x[0] == a and x[2] == b, compare_list))
            for args in curr_list:
                html_text += generate_html(*args)

            html_text += """
            </div>
        </div>
        </div>
    </section>
        """
            

    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(html_text)
        # print(generate_html(*args))
        # break

    