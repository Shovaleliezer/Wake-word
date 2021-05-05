# Wake-word
in this project I tried to create my own voice assistance 
how i did it? first i took my data from torchaudio SPEECHCOMMANDS
then i created helper function called label_to_index which gives me 1 if 
the word i read as a string is my keyword(keyword gets inside the function as param)
and 0 if its not the keyword. 
in order to get the data as i wish i created a function collate_fn that gets the batch and returns from the batch
only the data i want which is the wavefrom and the label
then i created normal train and eval functions and called transform function in order to have the same frequency
in all of my data, in the forward of my model i have created a normal 1d conv and called it 4 times with batch normalize
and max pooling and one fully connected at the end nothing special. 
i tuned some hyper parameters untill i got the accuracy i wanted which is over 99 percent and i checked also the precision and recall
(90% and 95%)
