# Root nodules classification

All the code belongs to the file title `nodules_identifier` which can be tested in the command line.

> nodules_Identifer.py -i --input_image_directory -t --meta_type [CSV|JSON] -f --image_type -m --input_metadata_directory -o --output_dir -h --help

To classify root nodules, first you need to pass the path of the directory that contains the images, in our case `images/`. Secondly, you must pass the path of the metadata file (a CSV or JSON), that represents the positions of the bounding boxes of the images, like the CSVs files included on the `meta/` directory. You also must inform the format of the root nodules images: PNG or JPG. Then, finally, you should choose a path to save the final results, which are images with the bounding boxes of the nodules drawed and CSVs that contains the predicions, where 0 represents a false-nodule and 1 a nodule.

You must know that the default classifier used on the predictions is the best algoritm tested by the researches, e.i., a fine-tuned Random Forest. Besides, like the default classifier, the dataset used to train the algoritm is hard-coded, so if you want to change it, you will need to write-out on the code.
