%% Image Resizing
imageResizingNeeded = true;

if imageResizingNeeded
    foldername = 'C:\Users\hujo8\OneDrive\Advanced image analysis\9 Week\TrainingData\';
    newdir = fullfile('C:\Users\hujo8\OneDrive\Advanced image analysis\9 Week\TrainingDataresized\');
    srcDir = fullfile(foldername);
    srcFiles = dir([srcDir,'*.JPG']);
    for i = 1 : length(srcFiles)
        filename = [srcDir, srcFiles(i).name];
        im = imread(filename);
        k = imresize(im,[416,416]);
        newfilename = [newdir,srcFiles(i).name];
        imwrite(k,newfilename,'JPG');
    end
end

%%

a = objectDetectorTrainingData(gTruthTrain,'SamplingFactor',1)






