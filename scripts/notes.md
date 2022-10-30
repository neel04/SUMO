# TEMP notes until `WandB`
- indexing per chunk ==> 1 Min.

`decode_jpeg` @ nexar_large_speed.py might be useful for decoding from TFRecords

- Use Gauge chart/speedometer chart for viz, make probability distribution colored and GT
  the arrow. Interpolate accordingly.

### Preparing TFRecords

- [x] Apparently, `read_one_video` gets stuck sometimes - Debug that | Doesn't reach FFMPEG snippet, because it doesn't proceed on the next video
- ^^ So it seems as long as one doesn't delete the AWS directory it works... no idea why, since it should already create an empty one (and does, but just freezes after that)

Create only a single file, 

```bash
clear; python3 scripts/BDD_Driving_Model/data_prepare/prepare_tfrecords.py --video_index='./data/bdd100k/video_filtered_38_60.txt' --output_directory='./data/bdd100k/tfrecords/train_1' --temp_dir_root=/home/awesome/mytemp
```

---

```bash
clear; python3 scripts/BDD_Driving_Model/data_prepare/prepare_tfrecords.py --video_index='./data/bdd100k/video_filtered_38_60.txt' --output_directory='s3://s-laion/ssd-videos/tfrecords/train_1' --temp_dir_root=/home/awesome/mytemp
```

| Workers      | Wall Time |
| ----------- | ----------- |
| 40      | 5m 30s (+- 20s)       |
| 200   | 5m 5s        |
## Shifting files
(Moves) Files: `train -> train_merged`
```bash
rsync -r --include='*.json' --exclude='*' ./train/ ./train_merged/ --remove-source-files
```

### AWS useful commands

empty out `TFRecords` folder, dryrun mode to verify
```bash
aws s3 rm s3://s-laion/ssd-videos/tfrecords --recursive --dryrun
```

Executing arbitary code in terminal, single-liner
```py
ipython -c "%run /opt/awesome/scripts/Final_converter.ipynb"
```

srun --partition=compute-od-cpu --nodelist=compute-od-cpu-dy-x2iezn-12xlarge-[1-100] --pty bash -i