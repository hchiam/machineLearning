https://youtu.be/Gj0iyo265bc?t=1m10s

https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/


# Key Terminal Commands:

```
git clone https://github.com/googlecodelabs/tensorflow-for-poets-2

cd tensorflow-for-poets-2

curl http://download.tensorflow.org/example_images/flower_photos.tgz \
    | tar xz -C tf_files

ls tf_files/flower_photos

IMAGE_SIZE=224

ARCHITECTURE="mobilenet_0.50_${IMAGE_SIZE}"

python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/flower_photos

python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/flower_photos/daisy/21652746_cc379e0eea_m.jpg

python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/flower_photos/roses/2414954629_3708a1a04d.jpg 
```
