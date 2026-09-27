| arm                                | reference      |   median_ratio_arm_over_ref |   ratio_ci_lo |   ratio_ci_hi |   mannwhitney_p |   cliffs_delta |
|:-----------------------------------|:---------------|----------------------------:|--------------:|--------------:|----------------:|---------------:|
| jvm_onnx_inprocess                 | py_onnxruntime |                      1.3026 |        1.2802 |        1.3244 |               0 |         0.6569 |
| jvm_rest_waitress_onnx             | py_onnxruntime |                     33.2456 |       32.0524 |       34.7261 |               0 |         0.9996 |
| jvm_rest_waitress_sklearn          | py_onnxruntime |                    202.504  |      198.407  |      206.738  |               0 |         1      |
| jvm_smile_inprocess                | py_onnxruntime |                      0.6053 |        0.5895 |        0.6228 |               0 |        -0.7483 |
| py_sklearn_njobs1                  | py_onnxruntime |                    174.21   |      170.343  |      177.48   |               0 |         1      |
| py_sklearn_njobsall                | py_onnxruntime |                    943.651  |      761.682  |     1200.44   |               0 |         1      |
| rest_flaskdev_njobsall_nokeepalive | py_onnxruntime |                   2123.83   |     2030.35   |     2208.98   |               0 |         1      |
| rest_waitress_onnx_keepalive       | py_onnxruntime |                     41.3596 |       40.5751 |       42.0312 |               0 |         1      |
| rest_waitress_sklearn_keepalive    | py_onnxruntime |                    201.02   |      197.24   |      204.082  |               0 |         1      |
| rest_waitress_sklearn_nokeepalive  | py_onnxruntime |                    222.614  |      218.519  |      225.812  |               0 |         1      |