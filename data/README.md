# Data

Original zips (`train_set.zip` ~497 MB, `test_set.zip` ~418 MB) exceed GitHub's 100 MB file limit and are **not tracked**.

- Local path (author's machine): `/Users/yangdaming/Downloads/Asus/train_set.zip` / `test_set.zip`
- Extracted structure expected by `src/train.py`:
  ```
  data/
  ├── 9_16_train_set/train/{EO,IR}/{image,label}/
  └── 9_30_test_set/test/{EO,IR}/image/
  ```
- Or set `DATA_ROOT` / `TEST_ROOT` in `src/train.py` and `src/inference.py` to your own paths.
- Colab one-click notebook handles download/unzip automatically: https://colab.research.google.com/drive/1t8fAJvJTvs8IVvYGcZohkxGd5TGmSHzd?usp=sharing

If you need the zips, contact the authors or download from the challenge platform.
