# Predictive_Maintenance

In this repo one can find the necessary tools for analysing vibration signal for applying predictive maintenance

Feature Extraction 

Necessary functions for chunking signal data.One can adjust the chunk size by changing the function parameters

Time and frequency domain features extraction function from the given signal data. 

Features extracted for each chunk and converted to dataframe.

Also labelling the signal can be handled by the function defined in this script.

Signal Classifier

Signal Classifier includes required function and the benchmark module for classifying the faults defined in signal.
This part can be adjusted by parameter tunning. 

Markov

Markovian analysis can be made by this script.
After classifying the states of the machine steady state probabilities.
