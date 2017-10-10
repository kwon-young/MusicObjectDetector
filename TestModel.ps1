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


$configuration_name = "500-epochs_many_anchor_box_scales"
Start-Transcript -path "$($pathToTranscript)2017-09-08_$($configuration_name)_test.txt" -append
python "$($pathToTranscript)TestModel.py" --path testdata --num_rois 50 --config_filename 2017-09-08_500-epochs_many_anchor_box_scales.pickle --network resnet50
Stop-Transcript
