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


################################################
# Upcoming Trainings 
################################################

# Started on Donki, 28.08.2017
Start-Transcript -path "$($pathToTranscript)2017-09-05_resnet50.txt" -append
python "$($pathToTranscript)TrainModel.py" --network resnet50 --output_weight_path "2017-09-05_resnet50.hdf5"
Stop-Transcript

#######################################################
# Below are configurations that already were 
# started on a machine and should not run again, 
# so we will terminate this PS-script here
# but retain those configurations for documentation
#######################################################
exit