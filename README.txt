xw2941 Xichen Wang
05/06/2024
"Exploring End-to-End Speech Recognition with LSTM Models- An Experimental Approach"

**Abstract**
This study explores the development and challenges associated with implementing an end-to-end (E2E) automatic speech recognition (ASR) system using the LibriSpeech dataset. The project investigates Long Short-Term Memory (LSTM) performance in terms of validation loss and Word Error Rate (WER). The study provides insights into the practical aspects of developing speech recognition technologies, highlighting the potential of E2E models to streamline the training process and improve accuracy without the need for intermediate phonetic transcription. 

**Tools**
Python

**Environment Set Up**
- I have set up a condo environment with required packages. Follow Steps to activate it.

1. First is to activate Anaconda, use following commands:

ls -d /home/xw2941/anaconda3/bin/
source /home/xw2941/anaconda3/bin/activate
conda activate py310

When you see (base) in front of your surname which means Anaconda is working!

2. Activate the specific condo env called 'asr':
conda activate asr

When you finish using conda, using following command to deactivate:
conda deactivate

3. Now you can start to run .py

If you only want to test Inference process, just follow **Inference**! Might take 5 minutes to finish.

**Inference**

Use command 'python main_test.py'

!!No need to change the path in config.py

This script will use a trained-model on test_dataset and output 'WER' and 'predicted sentence'.

- Sample output:

The output Sentence might be long, just move to the bottom and you can see the 'loss' and 'Were'.

  *Predictions: ['HE HOPED THERE WOULD BE STOW FOR DINNER TURNIPS AND CAROTS END BRUZED POTATOS AND FAT MUTTON PIECES TO BE LATLEED OUT IN THE THICK PEPPERD FLOER FATTAN SAUCE']
  *References: ['HE HOPED THERE WOULD BE STEW FOR DINNER TURNIPS AND CARROTS AND BRUISED POTATOES AND FAT MUTTON PIECES TO BE LADLED OUT IN THICK PEPPERED FLOUR FATTENED SAUCE']
  *Test Loss: 0.38169428110122683
  *Test WER: 0.30928978467893065

**Train**
If you want to go through the Training Process

Use command 'python main_train.py'

This script will automatically download the dataset and finish the whole train process

- Sample Output:
  1. A 'training_logs.csv'
  2. 'Model.pth'
  3. val_loss, val_wer

