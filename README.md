# Heartbeat classifier
Heartbeat classifier based on MIT-BIH Arrythmia Database using 1D-CNN to classify beat signals as normal, ventricular ectopic, superventricular ectopic or fused. 

## Database : MIT-BIH Arrythmia 
- 48 recordings of 30 min, each recording as two ECG signals
- 23 normal recordings (100's files) and 25 recordings with uncommon arrythmias (200's files), 4 paced recordings
- for each record, you have : a signal file (.dat), a header file (.hea), a annotation file (.ann)
- the signal file comprises the two signals, but the annotation file annotates the MLII signal (usually channel 0 of the signal file) with beat peak, beat type, ..

## Preprocessing
1. Separate patients in training and test sets. \\
Reproduction of the sets given in "Chazal, Philip & O'Dwyer, M. & Reilly, Richard. (2004). Automatic Classification of Heartbeats Using ECG Morphology and Heartbeat Interval Features.", meaning two sets of 22 patients (to be able to have all beat type in both sets). It is important to have all the beats from a single patient in a single set (ie either in training set or testing set), to avoid the NN to memorize patient-specific patterns, and therefore have better results on the testing sets since the patterns have already been seen. 

2. Create dictionnary from MIT-BIH annotations to AAMI standards \\
The classical equivalence as given in most articles is used here too. The unknown beats are discarded.

3. Split MLII signal into beat peaks \\
The annotation file is used to find the peaks. The interesting part of the signal around each peak is then extracted and keep as a training sample. The sample is labeled using the annotation file and the dictionnary previously created. The same window size is used for all the beats.

4. Manage class imbalance \\
Normal beats are overwhelmingly present in the dataset. Two solutions are tested to manage the imbalance. The first one is the use of a random over sampler to increase the number of samples of the minority classes to half of the number of samples of the majority class.

## Neural network
As the data to analye is an ensemble of signals, the NN used is a simple 1D-CNN one. In particular, the NN is made of 3 layers of conv+maxpooling, associated with a linear layer for classification. 

## Training 
A classical training approach is used, using PyTorch. The loss used is the cross entropy loss (classical for multiclass classification). The optimzer s Adam (adaptve learnng rate). Training is made over 30 epochs to minimize computing time while having relevant results. \\
Two methods are applied to deal with class imbalance : 
1. Keep the original data distribution but add class weights used in the computation of the loss. 
2. Use the previously created balanced sets and don't add class weights. 

## Evaluation
Since the classes are so unbalanced, accuracy alone doesn't show enough. Indeed, a network classifying all beats as "Normal" will have a high accuracy since most of the beats are "Normal". A mmore detailed classification report showing precision, recall, ... is used. And at the end of training, a confusion matrix is computed and displayed.

## Results
1. Training time\\
As expected, the method using the balanced sets take more time, indeed, there are simply more sample to analyze and therefore more computing to do. Using class weights seems to be the cheapest option. 

2. Final confusion matrix \\
- For the solution with class weights : the matrix is pretty relevant with most examples classified correctly. We can however see than the "Normal" prediction are more usual which is logical since there are way more "Normal" beats. The "S-beat" are as often classified as "Normal" beat as "S-beat", which seems logical since one of the difference between "S-beat" and "Normal" beat is that the "S-beat" starts earlier, which cannot be seen on windows centered around the peak. 
- For the solution without class weights : the matrix shows poor classification for "S-beat" and "F-beat". This is probably due to the fact than every minority class is increased to half the majority class. Improvement can probably be done by training a bit longer or by having the same number of examples in each class. However it would take way more longer. 
- 

## Improvements
Improvements can be done on the solution with class weights and patient separation by : 
- Adding layers to the network and training longer
- Adding other features to the input (such as the timing from the previous peak) and changing the network type (LSTM)












## References
- MIT-BIH Database : https://physionet.org/content/mitdb/1.0.0/ 
- Chazal, Philip & O'Dwyer, M. & Reilly, Richard. (2004). Automatic Classification of Heartbeats Using ECG Morphology and Heartbeat Interval Features. 


