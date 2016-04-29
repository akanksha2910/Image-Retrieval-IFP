Tools for training, testing, and compressing Fast R-CNN networks.

----

To visualize the bounding boxes, use the tool:

```
python tools/draw_proposals.py --network ./models/pascal_voc/VGG16/faster_rcnn_alt_opt/faster_rcnn_test.pt --model ./data/faster_rcnn_models/VGG16_faster_rcnn_final.caffemodel --image [Image File Name] --num_proposal 128
```
