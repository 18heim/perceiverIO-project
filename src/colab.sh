#!/bin/bash

# Install dependencies
echo "Install dependencies ..."
pip install -r ./requirements.txt
# Download dataset
echo "Press 'q' to exit"
echo "Press 'a' to download tiny-imagenet \n Press 'b' to download Sintel \n"
count=0
while : ; do
read -n 1 k <&1
if [[ $k = q ]] ; then
printf "\nQuitting from the program\n"
break
# Set up tiny-imagenet
elif [[ $k = a ]] ; then
printf "\nDownloading tiny-imagenet ...\n"
wget http://cs231n.stanford.edu/tiny-imagenet-200.zip -P /content/
unzip /content/tiny-imagenet-200.zip -d /content/
break
elif [[ $k = b ]] ; then
printf "\nDownloading sintel\n"
wget http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip -P /content/
unzip /content/MPI-Sintel-complete.zip -d /content/sintel/
break
else
((count=$count+1))
echo "Press 'q' to exit"
fi
done


