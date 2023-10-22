from torchvision import datasets
import csv
import pandas as pd
import numpy as np

# for the second Class
from torchtext.utils import (
    download_from_url,
)
from torchtext.data.datasets_utils import (
    _RawTextIterableDataset,
    _wrap_split_argument,
    _add_docstring_header,
    _create_dataset_directory,
    _create_data_from_csv,
)
import os
from string import punctuation


class ImageFolder2CSV:
    def __init__(self, ImagePath, CsvName, transform):
        """
            Args --> ImagePath: the path to the folder of the image
                 --> CsvName: the name for which you want to save the CSV file
                 --> transform: the transformations you want to apply on the data
        """

        self.ImagePath = ImagePath
        self.CsvName = CsvName
        self.transform = transform
    
    def change(self):
        """
            Converts and ImageFolder to a dataset
            Uses Pytorch ImageFolder --> refer to Pytorch Documentation

        """

        dataset = datasets.ImageFolder(self.ImagePath, transform=self.transform)
        return dataset


    def Image2Csv(self):
        """
        Takes in an Image dataset or an image of pixels and produces a csv file of pixels out of the image

        Args --> [dataset:  a dataset of pixels
                            Assumes the image is a gray scale image
                            example --> [1, 80, 80]
                Path: the path to which the Csv file is to be saved. type: string
        
        
        ]
        
        
        """

        with open(self.CsvName, 'w') as out:
            dataset = self.change()
            csv_out = csv.writer(out)
            # write to the pixel value rows
            #csv_out.writerow(columns)

            # iterate through the data to produce the splitted pixels
            for i in range(len(dataset)):
                splittedAndLabel = [dataset[i][1]]
                singleSub = dataset[i][0].flatten()
                for j in singleSub:
                    splittedAndLabel.append(j.item())
                    
                csv_out.writerow(splittedAndLabel)


class IterableDatasetForRawData:
    """
    Creates a raw Iterable dataset out of a normal csv
    Converts the csv the needed format
    
    Is Not Dataset Specific, Modify to meet needs

    """
    
    def read_csv(self, path):
        # read a csv file
        data = pd.read_csv(path) 

        # Converting the data to labels and data
        reviews = np.array(data['review']).tolist()
        labels = np.array(data['sentiment']).tolist()

        #modify to meet particular data
        reviews = [c for c in reviews if c not in punctuation]
        labels = [1 if label == 'positive' else 0 for label in labels]

        return reviews, labels
    
    def toCSV(self, path, relativePath):
        """
        Takes in a csv file and converts it to the needed format
        Args: 
            path -->  the name of the new csv
            relativePath --> the path to the old csv
        """

        # create a csv file
        reviews, labels = self.read_csv(relativePath)
        with open(path, 'w') as f:
            csv_out = csv.writer(f)
            for i in range(len(reviews)):
                csv_out.writerow([ str(labels[i]), reviews[i] ])
    
    def tryOut(self, split, path):
        NUM_LINES = {
                'train': 50000
                    }

        DATASET_NAME = "IMDB"


                # modify number of classes if your data has more than 2
        @_add_docstring_header(num_lines=NUM_LINES, num_classes=2)
        #@_create_dataset_directory(dataset_name=DATASET_NAME)
        @_wrap_split_argument(('train'))

        #Modify name according to the dataset
        def Data(root, split):
        
            return _RawTextIterableDataset(DATASET_NAME,NUM_LINES[split],
                                        _create_data_from_csv(path))
        return Data(split)

    
    


