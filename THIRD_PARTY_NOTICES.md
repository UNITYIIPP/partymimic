# Third-party notices

PartyMimic is derived from [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) (Fuyi Chen, MIT License, notice in
`LICENSE`). Portions of `src/partymimic/` and `tools/` come from
[supermimic_vibed](https://github.com/zita-ch/supermimic_vibed) (Chong Zhang) and
[InstantTrack](https://github.com/Huy-Quang-Dao/instantTrack) (Huy-Quang-Dao), used under their MIT Licenses,
reproduced below.

## InstantTrack

```
MIT License

Copyright (c) 2026 Huy-Quang-Dao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## supermimic_vibed

```
MIT License

Copyright (c) 2026 ETH Zurich, Chong Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Dependencies and data

The Python dependencies (mjlab, MuJoCo, MuJoCo Warp, NVIDIA Warp, PyTorch, matplotlib, Pillow) are installed from
their own distributions under their own licenses (Apache-2.0, BSD-3-Clause and others) and are not redistributed here.
The Unitree G1 description (BSD-3-Clause) is downloaded by `tools/download_data.sh`.

The only motion clips in this repository are the 56 kneeling and crawling clips in `assets/motions/kneel_crawl` (G1
and their RP1 retargets). They are ours and released under this repository's MIT License. All other motion data are
downloaded from their sources by `tools/download_data.sh` and `tools/data/build_library.sh` and are not redistributed:

- LAFAN1 (Ubisoft La Forge), as the G1 retarget of
  [lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset):
  CC BY-NC-ND 4.0.
- BONES-SEED ([bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed), gated):
  [BONES-SEED license](https://bones.studio/info/seed-license).
- KungfuAthlete ([LuluCao/KungfuAthleteBot](https://huggingface.co/datasets/LuluCao/KungfuAthleteBot),
  [NPCLEI/KungFuAthleteBot](https://github.com/NPCLEI/KungFuAthleteBot)): Apache-2.0.
- XtremeMotion, the curated motion subset linked from the README of
  [Perkins729/OmniXtreme](https://github.com/Perkins729/OmniXtreme): Apache-2.0, the license of that repository (the
  file carries no separate terms).
- MotionDecode ([CMRobot/MotionDecode](https://huggingface.co/datasets/CMRobot/MotionDecode), test sets only):
  non-commercial access terms.

Files derived from these datasets (the videos in `docs/static/videos`, checkpoints, built motion libraries) are not
covered by this repository's MIT License and follow the licenses of the data they come from.

## EVT integration

The EVT control values in `assets/evt/config.json` come from the user's local allmimic checkout, branch `armature`,
commit `e17f339` (`DEX_EVT_CFG`). `tools/prepare_evt.py` imports its `assets/robots/evt/urdf/evt2_simple.xml` and
STL meshes into the git-ignored `data/evt/` directory and records source hashes. These third-party robot assets
are not redistributed here, and this repository does not assign them a new license. The EVT LAFAN1 motion archive
is read from its existing location; the LAFAN1 data terms above still apply.

The optional `assets/evt_nominal/config.json` records nominal PD values from the local allmimic commit
`01e6ad299e7db753a7822551dbd58c00548a8c30` and passive dynamics/effective motor limits from its compiled
`assets/robots/evt/urdf/evt2_simple.xml`. Source hashes are included in the configuration. Its generated robot
assets are kept separately in the git-ignored `data/evt_nominal/`, under the same asset terms described above.
