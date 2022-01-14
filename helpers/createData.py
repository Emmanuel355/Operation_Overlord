from torchvision import datasets
import csv


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