$pathToSourceRoot = "C:\Users\Alex\Repositories\MusicObjectDetector\"
$pathToTranscript = "$($pathToSourceRoot)"

# Allowing wider outputs https://stackoverflow.com/questions/7158142/prevent-powergui-from-truncating-the-output
$pshost = get-host
$pswindow = $pshost.ui.rawui
$newsize = $pswindow.buffersize
$newsize.height = 9999
$newsize.width = 1500
$pswindow.buffersize = $newsize

cd $pathToSourceRoot
echo "Appending source root $($pathToSourceRoot) to temporary PYTHONPATH"
$env:PYTHONPATH = $pathToSourceRoot

#python C:\Users\Alex\Repositories\MusicSymbolClassifier\ModelTrainer\tests\Symbol_test.py
#cd "tests"
#pytest


#######################################
# Build tools for data_generators_fast
#######################################
cd keras_frcnn
cd py_faster_rcnn
python setup.py build_ext --inplace
cd ..
cd ..

################################################
# Upcoming Trainings 
################################################

# Train two vgg networks with more boxes and more rois
$model_name = "vgg"

$configuration_name = "small_anchor_box_scales_many_rois"
$base_name = "2017-11-07_600-rpns_0.7-overlap"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 600 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript

$base_name = "2017-11-08_800-rpns_0.7-overlap"
$configuration_name = "small_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 800 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript


# Play with max-boxes
$model_name = "resnet50"

$base_name = "2017-11-09_800-rpns_0.7-overlap"
$configuration_name = "small_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 800 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript

$base_name = "2017-11-10_1000-rpns_0.7-overlap"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 1000 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript

$base_name = "2017-11-11_1200-rpns_0.7-overlap"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 1200 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript

# Play with overlap thresholds
$base_name = "2017-11-12_600-rpns_0.6-overlap"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 600 --non_max_suppression_overlap_threshold 0.6
Stop-Transcript

$base_name = "2017-11-13_600-rpns_0.8-overlap"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 600 --non_max_suppression_overlap_threshold 0.8
Stop-Transcript

# Play with different configurations
$configuration_name = "small_anchor_box_scales"
$base_name = "2017-11-14_600-rpns_0.7-overlap"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 600 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript

$configuration_name = "small_anchor_box_scales_many_rois"
$base_name = "2017-11-15_600-rpns_0.7-overlap"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 600 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript

$configuration_name = "small_images"
$base_name = "2017-11-16_600-rpns_0.7-overlap"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 600 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript

$configuration_name = "stretched_anchor_box_ratios"
$base_name = "2017-11-17_600-rpns_0.7-overlap"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 600 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript

$configuration_name = "many_anchor_box_scales"
$base_name = "2017-11-18_600-rpns_0.7-overlap"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 600 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript

$configuration_name = "many_anchor_box_ratios"
$base_name = "2017-11-19_600-rpns_0.7-overlap"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 600 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript

$configuration_name = "many_anchor_box_scales_many_rois_small_stride"
$base_name = "2017-11-20_600-rpns_0.7-overlap"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 600 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript

$configuration_name = "many_anchor_box_scales_many_rois_medium_stride"
$base_name = "2017-11-21_600-rpns_0.7-overlap"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 600 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript

$configuration_name = "many_anchor_box_scales_many_rois_large_stride"
$base_name = "2017-11-22_600-rpns_0.7-overlap"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 600 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript



#######################################################
# Below are configurations that already were 
# started on a machine and should not run again, 
# so we will terminate this PS-script here
# but retain those configurations for documentation
#######################################################
exit

# Started on Donkey, 03.11.2017
$base_name = "2017-11-03-600rpns"
$model_name = "vgg"
$configuration_name = "small_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 600 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript

$model_name = "resnet50"
$configuration_name = "small_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200 --non_max_suppression_max_boxes 600 --non_max_suppression_overlap_threshold 0.7
Stop-Transcript

# Started on Donkey, 31.10.2017
$base_name = "2017-10-31_1epoch"
$model_name = "resnet50"
$configuration_name = "small_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 1
Stop-Transcript

$base_name = "2017-10-31"
$model_name = "vgg"
$configuration_name = "small_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200
Stop-Transcript

$base_name = "2017-10-31"
$model_name = "resnet50"
$configuration_name = "small_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($model_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --model_name $model_name --configuration_name $configuration_name --output_weight_path "$($base_name)_$($model_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($model_name)_$($configuration_name).pickle" --num_epochs 200
Stop-Transcript

# Started on Donkey, 30.10.2017
$base_name = "2017-10-30_500-epochs"
$number_of_epochs = 500
$configuration_name = "small_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

# Started on Donkey, 07.09.2017
$base_name = "2017-09-07_5-epochs"
$number_of_epochs = 5
$configuration_name = "streched_anchor_box_ratios"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

$configuration_name = "small_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

$configuration_name = "many_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

$configuration_name = "small_images"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

$configuration_name = "many_anchor_box_scales_many_rois_large_stride"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

$configuration_name = "many_anchor_box_scales_many_rois_medium_stride"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

$configuration_name = "many_anchor_box_scales_many_rois_small_stride"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript


$base_name = "2017-09-08_500-epochs"
$number_of_epochs = 500
$configuration_name = "streched_anchor_box_ratios"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

$configuration_name = "small_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

$configuration_name = "many_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

$configuration_name = "many_anchor_box_scales_many_rois_small_stride"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

$configuration_name = "many_anchor_box_scales_many_rois_medium_stride"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript


$configuration_name = "many_anchor_box_scales_many_rois_large_stride"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript


$base_name = "2017-09-09_1000-epochs"
$number_of_epochs = 1000
$configuration_name = "streched_anchor_box_ratios"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

$configuration_name = "small_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

$configuration_name = "many_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

$configuration_name = "many_anchor_box_scales_many_rois_small_stride"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript

$configuration_name = "many_anchor_box_scales_many_rois_medium_stride"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript


$configuration_name = "many_anchor_box_scales_many_rois_large_stride"
Start-Transcript -path "$($pathToTranscript)$($base_name)_$($configuration_name).txt" -append
python "$($pathToTranscript)TrainModel.py" --configuration_name $configuration_name --output_weight_path "$($base_name)_$($configuration_name).hdf5" --config_filename "$($base_name)_$($configuration_name).pickle" --num_epochs $number_of_epochs
Stop-Transcript



# Started on Donki, 05.09.2017
Start-Transcript -path "$($pathToTranscript)2017-09-05_resnet50.txt" -append
python "$($pathToTranscript)TrainModel.py" --recreate_dataset_directory --network resnet50 --output_weight_path "2017-09-05_resnet50.hdf5" --config_filename "2017-09-05_config.pickle"
Stop-Transcript
