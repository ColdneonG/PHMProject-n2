# Experimental results

Mean ± sample standard deviation across seeds 42, 3407 and 2026. Computed from the per-run metrics included in this repository.

## Main cohort

| Model | Dice | IoU | Small Dice | Boundary F1 | Level MAE | Blur Dice |
| --- | --- | --- | --- | --- | --- | --- |
| deeplabv3plus_resnet18_pretrained | 0.7515 ± 0.0473 | 0.6335 ± 0.0583 | 0.5129 ± 0.0591 | 0.2909 ± 0.0462 | 0.1057 ± 0.0126 | 0.7013 ± 0.0121 |
| localsk_conv_resnet18_pretrained | 0.7491 ± 0.0195 | 0.6394 ± 0.0131 | 0.4831 ± 0.0961 | 0.2825 ± 0.0216 | 0.1066 ± 0.0115 | 0.6954 ± 0.0788 |
| localsk_hc_global_conv_resnet18_pretrained | 0.7823 ± 0.0249 | 0.6744 ± 0.0286 | 0.4781 ± 0.0938 | 0.2990 ± 0.0194 | 0.1080 ± 0.0053 | 0.7394 ± 0.0413 |
| localsk_hc_resnet18_pretrained | 0.7212 ± 0.0231 | 0.6122 ± 0.0115 | 0.6210 ± 0.0316 | 0.2621 ± 0.0192 | 0.1249 ± 0.0372 | 0.6857 ± 0.1504 |
| phm_project_n2 | 0.7864 ± 0.0433 | 0.6700 ± 0.0558 | 0.6663 ± 0.0186 | 0.2665 ± 0.0205 | 0.0977 ± 0.0113 | 0.7747 ± 0.0240 |
| pidnet_s | 0.6739 ± 0.0371 | 0.5492 ± 0.0332 | 0.3844 ± 0.1070 | 0.2366 ± 0.0264 | 0.1505 ± 0.0235 | 0.7443 ± 0.0348 |
| pspnet_resnet18_pretrained | 0.6436 ± 0.0317 | 0.5454 ± 0.0330 | 0.3586 ± 0.0099 | 0.2322 ± 0.0168 | 0.1254 ± 0.0095 | 0.5452 ± 0.1478 |
| segformer_b0_pretrained | 0.7068 ± 0.0379 | 0.5935 ± 0.0432 | 0.5453 ± 0.1643 | 0.2878 ± 0.0319 | 0.1383 ± 0.0374 | 0.6935 ± 0.0854 |
| unet_resnet18_pretrained | 0.6846 ± 0.0421 | 0.5948 ± 0.0412 | 0.4524 ± 0.0813 | 0.2945 ± 0.0176 | 0.1190 ± 0.0404 | 0.6158 ± 0.0940 |

## Supplementary cohort

| Model | Dice | IoU | Small Dice | Boundary F1 | Level MAE | Blur Dice |
| --- | --- | --- | --- | --- | --- | --- |
| concat_hc_global_conv_resnet18_pretrained | 0.7598 ± 0.0128 | 0.6442 ± 0.0124 | 0.4726 ± 0.1060 | 0.2862 ± 0.0068 | 0.1124 ± 0.0192 | 0.7342 ± 0.0041 |
| localsk_hc_global_conv_resnet18_pretrained | 0.7151 ± 0.0870 | 0.6006 ± 0.0832 | 0.5038 ± 0.1398 | 0.2677 ± 0.0372 | 0.1247 ± 0.0322 | 0.7381 ± 0.0170 |
| localsk_hc_global_group2_resnet18_pretrained | 0.7211 ± 0.0728 | 0.6235 ± 0.0775 | 0.4906 ± 0.0465 | 0.2786 ± 0.0134 | 0.1087 ± 0.0441 | 0.6821 ± 0.0804 |
| localsk_hc_global_insk_conv_resnet18_pretrained | 0.7466 ± 0.0314 | 0.6336 ± 0.0310 | 0.5253 ± 0.0435 | 0.2610 ± 0.0183 | 0.1089 ± 0.0176 | 0.7441 ± 0.0662 |
| localsk_hc_global_lowrank85_resnet18_pretrained | 0.7406 ± 0.0157 | 0.6211 ± 0.0177 | 0.5999 ± 0.1538 | 0.2578 ± 0.0066 | 0.1236 ± 0.0092 | 0.6867 ± 0.0951 |
| matched_a0_standard_aspp_resnet18_pretrained | 0.7291 ± 0.0225 | 0.6221 ± 0.0208 | 0.5984 ± 0.0557 | 0.2617 ± 0.0450 | 0.1107 ± 0.0306 | 0.7334 ± 0.0885 |
| mean_hc_global_conv_resnet18_pretrained | 0.7535 ± 0.0499 | 0.6398 ± 0.0523 | 0.5314 ± 0.0332 | 0.2808 ± 0.0175 | 0.1015 ± 0.0086 | 0.7574 ± 0.0222 |
| phm_project_n1 | 0.7333 ± 0.0417 | 0.6271 ± 0.0413 | 0.4869 ± 0.1440 | 0.2564 ± 0.0082 | 0.1215 ± 0.0093 | 0.7414 ± 0.0471 |
| phm_project_n2 | 0.7359 ± 0.0451 | 0.6124 ± 0.0546 | 0.5640 ± 0.0560 | 0.2682 ± 0.0080 | 0.1315 ± 0.0178 | 0.6867 ± 0.0570 |
| phm_project_n4 | 0.7207 ± 0.0890 | 0.6091 ± 0.0954 | 0.6114 ± 0.0571 | 0.2732 ± 0.0293 | 0.1122 ± 0.0444 | 0.6161 ± 0.1159 |
| sum_hc_global_conv_resnet18_pretrained | 0.7545 ± 0.0293 | 0.6332 ± 0.0353 | 0.5856 ± 0.1111 | 0.2489 ± 0.0005 | 0.1084 ± 0.0094 | 0.7093 ± 0.0578 |

## Main-cohort comparison with DeepLabV3+

Blur Dice improvement: 7.34 percentage points. Blur Level MAE relative reduction: 19.98%.

The supplementary cohort is reported separately, including its repeated A3/A4 anchors. Each series is summarized over its own three seeds.
