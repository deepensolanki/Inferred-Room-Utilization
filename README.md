# Inferred-Room-Utilization

Indirect room occupancy measurement using Raspberry Pi and Neural Networks

Complete information here https://www.deepensolanki.com/articles/inferred-room-utilization

Temperature, Humidity and CO2 measurements were recorded for 3 days while changing the number of people in the room for the training data set. 

*****************************************************************************************************************************************
ALL THE DATA COLLECTED WAS FROM A SINGLE LOCATION BECAUSE OF THE COVID19 SHELTER IN PLACE ORDER 
MORE DATA CAN BE COLLECTED AND TRAINED WITH THIS MODEL
******************************************************************************************************************************************

A bi-directional LSTM was trained to recognize the effect of adding or removing people from a room. The model looks at the signatures
of temperature, humidity and CO2 for 5 minutes and outputs a number between -6 and +6 denoting how many people were either added or subtracted from the room. To include the model for more number of people it must be retrained with the 'n' value changed. The file also contains directions for how to write the main loop. The "delta based approach" is aimed at generalizing the usage of this method. 
