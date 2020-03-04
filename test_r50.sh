 #!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./tools/dist_test.sh configs/solo/r50_p6.py work_dirs/r50_p6/latest.pth 8  --json_out ret_10
