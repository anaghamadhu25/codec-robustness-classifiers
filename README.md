# Effects of JPEG, WebP and AVIF compression on CNN and Transformer image classifiers

Analysis code and per-image results for the paper
"Effects of JPEG, WebP and AVIF Compression on CNN and Transformer Image Classifiers: A Statistical and Per-Class Analysis" (Anagha Madhu).

## What this repository contains

| File | Contents |
|---|---|
| `compression_vs_accuracy.ipynb` | The Colab notebook used for all experiments. It also contains exploratory cells (fine-tuning and BatchNorm recalibration attempts) that are not part of the paper. |
| `sample_list.csv` | The 1,000 sampled Imagenette validation images (100 per class, seed 42): file path, ImageNet class id, ImageNet-1k index, class name, width, height, pixel count. Row order defines `img_idx` (0 to 999) used in the other files. |
| `baseline_accuracy.csv`, `baseline_per_image.csv` | Accuracy and per-image correctness on the original images for ResNet-50 and ViT-B/16. |
| `all_results.csv` | One row per image, codec and quality setting (18,000 rows): `img_idx`, predictions and correctness for both models, file size, bits per pixel, PSNR, SSIM. |
| `summary_table.csv`, `accuracy_with_ci.csv` | Accuracy, bitrate and similarity per codec and quality, with 95% Wilson intervals. |
| `stats_vs_baseline.csv`, `stats_codec_comparison.csv`, `stats_model_comparison.csv` | Holm-corrected exact McNemar tests (compressed vs original, codec vs codec at similar bitrate, ViT vs ResNet). |
| `per_class_q20.csv`, `compression_induced_confusions.csv` | Per-class results at quality 20 and the ResNet-50 images that became wrong after compression, with their predicted classes. |

## Data

The images are not included. They come from Imagenette (320-pixel version, validation split), a subset of ImageNet, and remain subject to the terms of the original datasets:
https://github.com/fastai/imagenette
Download used: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
`sample_list.csv` lists the exact files used. By file name, 972 of the 1,000 images come from the ImageNet-1k training set (see the paper's Limitations).

## Models and software

- ResNet-50: `resnet50.a1_in1k`; ViT-B/16: `vit_base_patch16_224.augreg2_in21k_ft_in1k` (both from timm)
- Python 3.13, PyTorch 2.11.0, timm 1.0.30, Pillow 11.3.0, pillow-avif-plugin 1.6.0, scikit-image 0.25.2, SciPy 1.16.3, statsmodels 0.15.0, pandas 2.2.3
- Run on Google Colab with a Tesla T4 GPU

## Reproducing

Open the notebook in Google Colab, run the setup cells, then the experiment cells in order. The notebook was developed in stages, so after a session reset you may need to rerun earlier setup cells before later ones. AVIF encoding is the slowest step: in our session, compressing and saving 3,500 images took about 2 minutes with JPEG, 4 with WebP and 9 with AVIF.

File sizes, and therefore bitrates, can differ slightly with other encoder versions, so exact numbers may vary if library versions differ from those above.
