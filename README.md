<p align="center">
<h1 align="center"<strong>PhysGraph: Physically-Grounded Graph-Transformer Policies for Bimanual Dexterous Hand–Tool–Object Manipulation</strong></h1>
  <p align="center">
    <a href='https://blarklee.github.io/' target='_blank'>Runfa Blark Li</a>
    ·
    <a href='' target='_blank'>David Kim</a>
    ·
    <a href='' target='_blank'>Xinshuang Liu</a>
    ·
    <a href='' target='_blank'>Keito Suzuki</a>
    ·
    <a href='' target='_blank'>Dwait Bhatt</a>
    ·
    <a href='' target='_blank'>Nikola Raicevic</a>
    ·
    <a href='' target='_blank'>Xin Lin</a>
    ·
    <a href='' target='_blank'>Ki Myung Brian Lee</a><br>
    ·
    <a href='' target='_blank'>Nikolay Atanasov</a>
    ·
    <a href='' target='_blank'>Truong Nguyen</a>
    <br>
    UC San Diego
    <br>
  </p>
</p>
<p align="center">
  <a href='https://arxiv.org/abs/2603.01436'>
    <img src='https://img.shields.io/badge/arXiv-2503.19901-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
  <a href='https://arxiv.org/abs/2603.01436'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'></a>
  <a href='https://blarklee.github.io/PhysGraph_website_official/'>
    <img src='https://img.shields.io/badge/Project-Page-green?style=flat&logo=Google%20chrome&logoColor=green'></a>
</p>


## 📹 Demo
<p align="center">
    <img src="assets/artimano_all_labeled.gif" align="center" width=60% >
    <br>
</p>
While recent learning-based approaches have made substantial progress on dexterous manipulation, bimanual tool-use remains particularly challenging. PhysGraph significantly outperforms SOTA baseline on challenging bimanual tool-use tasks in success rate and motion fidelity, supports zero-shot generalization to unseen tool/object in different tasks, and is embodiment-agnostic to popular robotic dex-hands (Shadow, Allegro, Inspire)


<!-- teaser image -->
## 🏠 Overview
<p align="center">
    <img src="assets/pipeline.png" alt="teaser" width="100%">
</p>
PhysGraph is a physically-grounded graph-transformer policy designed explicitly for bimanual tool-object manipulation. Rather than flattening the state as concurrent dexhand manipulations, we formulate the bimanual system as a dynamic kinematic graph, where nodes represent individual rigid bodies (links, tools, objects) and edges represent physical couplings. Our approach introduces two key innovations: (i) We propose a per-link tokenization strategy. Instead of pooling states into a global embedding, we process each link’s multi-modal states as a distinct token, preserving fine-grained local properties. (ii) Most crucially, we introduce a novel Physically-Grounded Bias Generator. Unlike generic Graph Transformers (e.g., Graphormer) that utilizes abstract static graph distances for chemical bonds, we inject a dynamic learning-based head-specific composite bias directly into the attention mechanism. The composite bias includes Spatial Bias (kinematic chain distance), Dynamic Edge Bias (static/dynamic contact states), Geometric Bias (Cartesian proximity), and Anatomical Priors (serial/synergies kinematics), which enable our policy to explicitly reason about the physical connectivity and contact logic, focusing attention on contacting fingers or coordinated joints, thereby improving the reliability and precision.

## 📑 Table of Contents
1. [Installation](#Installation)
2. [Prerequisites](#Prerequisites)
3. [Usage](#usage)
3. [Citation](#Citation)
5. [Acknowledgement](#acknowledgement)

---

## 🛠️ Installation
<a id="Installation"></a>

<details>
<summary>Steps:</summary>

1. Clone the repository and initialize submodules:
    ```bash
    git clone https://github.com/BlarkLee/PhysGraph.git
    git submodule init && git submodule update
    ```
2. Create a virtual environment named `physgraph` with Python 3.8. Note that IsaacGym only supports Python versions up to 3.8.
    ```bash
    conda create -y -n physgraph python=3.8
    conda activate physgraph
    pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
    ```
3. Download IsaacGym Preview 4 from the [official website](https://developer.nvidia.com/isaac-gym) and follow the installation instructions in the documentation. Test the installation by running an example script, such as `joint_monkey.py`, located in the `python/examples` directory.
4. Install additional dependencies.
    ```bash
    pip install git+https://github.com/ZhengyiLuo/smplx.git
    pip install git+https://github.com/KailinLi/bps_torch.git
    pip install fvcore~=0.1.5
    pip install --no-index --no-cache-dir pytorch3d==0.7.3 -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu117_pyt1131/download.html
    pip install -r requirements.txt
    pip install -e . # include the current directory in the Python path. Or use: `export PYTHONPATH=.:$PYTHONPATH`
    pip install numpy==1.23.0 # downgrade numpy to 1.23.0 to avoid compatibility issues
    ```

</details>

---

## 📋 Prerequisites
<a id="Prerequisites"></a>

We follow the prerequisit of [ManipTrans](https://maniptrans.github.io/) to prepare the dataset.

<details>
<summary>Steps:</summary>

### `OakInk-V2` dataset
1. Download the OakInk-V2 dataset from its [official website](https://oakink.net/v2/) and extract it into the `data/OakInk-v2` directory. (You may skip downloading images; only annotated motion data is required.)

2. For each object mesh in `data/OakInk-v2/object_preview/align_ds`, generate the [COACD](https://github.com/SarahWeiii/CoACD) file by running:
    ```bash
    python physgraph_envs/lib/utils/coacd_process.py -i data/OakInk-v2/object_preview/align_ds/xx/xx.obj -o data/OakInk-v2/coacd_object_preview/align_ds/xx/xx.obj --max-convex-hull 32 --seed 1 -mi 2000 -md 5 -t 0.07
    # Or, if you have the ply file, you can use:
    python physgraph_envs/lib/utils/coacd_process.py -i data/OakInk-v2/object_preview/align_ds/xx/xx.ply -o data/OakInk-v2/coacd_object_preview/align_ds/xx/xx.ply --max-convex-hull 32 --seed 1 -mi 2000 -md 5 -t 0.07
    ```
3. For each generated COACD file in `data/OakInk-v2/coacd_object_preview/align_ds`, create a corresponding URDF file based on `assets/obj_urdf_example.urdf`.

4. Download the `body_upper_idx.pt` file from the [official website](https://oakink.net/v2/) and place it in the `data/smplx_extra` directory.

5. The directory structure should look like this:
    ```
    data
    ├── smplx_extra
    │   └── body_upper_idx.pt
    └── OakInk-v2
        ├── anno_preview
        ├── coacd_object_preview
        ├── data
        ├── object_preview
        └── program
    ```

</details>


## ▶️ Usage
<a id="usage"></a>
### BiManual Tool-Use Policies

1. **Preprocessing**

    Preprocess data for both hands:
    ```bash
    # for Artimano Hand
    python main/dataset/mano2dexhand.py --data_idx 083f7@0 --side right --dexhand artimano --headless --iter 7000
    python main/dataset/mano2dexhand.py --data_idx 083f7@0 --side left --dexhand artimano --headless --iter 7000
    # for other hands, just replace `Artimano` with the corresponding hand name. Candidate hand names are `Shadow`, `Inspire`, `Allegro`. 
    ```
    Regarding `data_idx` of OakInk V2, for example, `083f7@0` refers to the primitive task indexed at `0` in the sequence labeled `scene_01__A001++seq__083f7a577484ba7929a9__2023-04-27-19-25-24` (for simplification, we only use the first 5 digits of the hash code).

    To reuse an old trajectory with a new colored tool mesh and translate the
    right-hand grasp from the old red-region center to the new one, see
    [Geometric contact trajectory retargeting](docs/contact_trajectory_retargeting.md).

    For the Stage-1 red/black semantic signals, imitation-dominant 30% curriculum,
    semantic AdaLN-Zero, and sparse geometric attention bias, see
    [Stage-1 semantic contact reward and bias](docs/stage1_semantic_contact_reward_bias.md).

    For test-only operation completion, planned semantic contact coverage, and
    semantic success evaluated from saved checkpoints over the full operation, see
    [Checkpoint semantic contact evaluation](docs/checkpoint_semantic_contact_evaluation.md).

2. **Training**
  Artimano, Allegro, Inspire and Shadow share the direct-action Transformer;
  each hand uses its own retarget trajectories and PPO checkpoint. No pretrained
  single-hand imitator is required. See [four-hand direct-policy setup](docs/multi_hand_direct_policy.md).

  Train bi-manual policies:
    ```bash
    python main/rl/train.py task=ResDexHand dexhand=artimano side=BiH headless=true num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true dataIndices=[083f7@0] early_stop_epochs=10000 actionsMovingAverage=0.4 experiment=083f7@0_artimano
    ```
    The `early_stop_epochs` parameter can be adjusted based on the task complexity.
    Training also supports multi-GPUs, here is an example of 2-GPUs parallel training:
    ```bash
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 main/rl/train.py   task=ResDexHand dexhand=artimano side=BiH headless=true multi_gpu=true num_envs=4096 learning_rate=2e-4 test=false randomStateInit=true dataIndices=[083f7@0] early_stop_epochs=10000 actionsMovingAverage=0.4 experiment=083f7@0_artimano_2gpu
    ```
    To resume from a checkpoint, set `from_ckpt_epoch=true` and `checkpoint=xxx.pth` 
    
  </details>

3. **Test**
  Test the bi-manual policy:
    ```bash
    python main/rl/train.py task=ResDexHand dexhand=artimano side=BiH headless=false num_envs=4 learning_rate=2e-4 test=true randomStateInit=false dataIndices=[083f7@0] actionsMovingAverage=0.4 checkpoint=runs/083f7@0_artimano__xxxxxx/nn/083f7@0_artimano.pth
    ```

  For `task=ResDexHand side=BiH`, append any of these display overrides to the
  test command:

    ```bash
    test_visualization.left_hand_color='[0.3,0.6,1.0]' \
    test_visualization.right_hand_color='[1.0,0.6,0.3]' \
    test_visualization.object_color='[0.7,0.7,0.7]' \
    test_visualization.tool_color='[0.8,0.2,0.2]' \
    test_visualization.show_mano=false \
    test_visualization.contact_highlight=false
    ```

  These options apply only with `test=true`; training ignores them. Colors are
  RGB values in `[0, 1]`, with `null` preserving the existing appearance. A custom
  color replaces that actor's display texture with white for a solid color;
  source assets, collision geometry and semantic contact labels stay unchanged.
  `object_color` selects the left-hand task object; `tool_color` selects the
  right-hand tool, including a `new_obj` replacement. Both colors and textures
  apply to the viewer and camera images. By default, both switches are `true`
  and all colors are `null`, preserving the existing behavior. In the viewer,
  contact highlighting temporarily makes contacting hand links pink and restores
  their custom color (or the legacy white) after contact ends. Set
  `contact_highlight=false` for fixed hand colors or the original hand materials.
  `show_mano` controls both hands' reference skeleton lines in the viewer
  (`headless=false`); camera recordings do not include these lines.

  To replay retarget results directly (TACO and OakInk-V2 are both supported):

    ```bash
    python scripts/visualize_taco_retarget.py --data_idx '083f7@0' --dexhand artimano \
      --left_hand_color 0.3,0.6,1.0 --right_hand_color 1.0,0.6,0.3 \
      --object_color 0.7,0.7,0.7 --tool_color 0.8,0.2,0.2
    ```

  Each of the four `--*_color` arguments accepts comma-separated RGB numbers in
  `[0, 1]` for a solid color, or `urdf` to prefer URDF materials without the
  viewer's fixed color override (for example, `--tool_color urdf`). Omitted color
  arguments retain the existing colors and textures. Use `--hide_left_hand`,
  `--hide_right_hand`, `--hide_object`, and `--hide_tool` independently to hide
  parts at startup. `object` is the left-hand target and `tool` is the right-hand
  tool, including `--new_obj` replacements. All parts are visible by default.
  The replay still loads and validates both retarget trajectories; hiding parts
  only changes the viewer scene. The camera centers on visible parts, or the
  table when all four are hidden. Space pauses/resumes, R restarts, and Esc exits.
---



## Citation
<a id="Citation"></a>
```
@misc{physgraph,
      title={PhysGraph: Physically-Grounded Graph-Transformer Policies for Bimanual Dexterous Hand-Tool-Object Manipulation}, 
      author={Runfa Blark Li and David Kim and Xinshuang Liu and Keito Suzuki and Dwait Bhatt and Nikola Raicevic and Xin Lin and Ki Myung Brian Lee and Nikolay Atanasov and Truong Nguyen},
      year={2026},
      eprint={2603.01436},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2603.01436}, 
}
```

## 🙏 Acknowledgement
<a id="acknowledgement"></a>
We thank [OakInk V2](https://oakink.net/v2/) for the dataloader and [ManipTrans](https://maniptrans.github.io/) for the training pipeline used in this work.


# PhysGraph_CR
