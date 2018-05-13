import os
import pandas as pd
import numpy as np
from random import sample
from PIL import Image

def ensure_matching(data_dir):
    """
        Deletes rows from the attributes file that don't have a corresponding
        image in the directory of face images...deletes face images that don't
        have a corresponding row in the attributes file.
    """

    # Read in attributes file and sort by person names
    lfw_attrs = pd.read_csv(data_dir + '/lfw_attributes.txt', sep='\t', header=1)
    lfw_attrs = lfw_attrs.drop(columns=[
            'imagenum',
            'Wearing Necklace',
            'Wearing Earrings',
            'Wearing Necktie',
            'Black Hair',
            'Blond Hair',
            'Brown Hair',
            'Gray Hair',
            'Curly Hair',
            'Wavy Hair',
            'Straight Hair',
            'Outdoor',
            'Receding Hairline',
            'Wearing Hat',
            'Bangs',
            'Bald',
            'Harsh Lighting',
            'Soft Lighting',
            'Flash',
            ])
    lfw_attrs = lfw_attrs.reindex(columns=list(lfw_attrs.columns)+['Female'])
    lfw_attrs = lfw_attrs.sort_values(['person'])

    # Read in cropped lfw dataset and sort by file (person) names
    images = sorted(os.listdir(data_dir + '/faces'))

    # Helper to convert file name into person name
    def reformat(file_name):
        file_name = ''.join(file_name.split('.ppm')[:-1])
        return ' '.join(file_name.split('_')[:-1])

    # Copy attributes and images for processing
    imgs = list(images)
    attrs = list(lfw_attrs.iterrows())

    # Iterate through copied lists
    for index, ((_, attr), img) in enumerate(zip(attrs, imgs)):
        attr['Female'] = -attr['Male']
        if attr['person'] != reformat(img):
            # Attribute file's person does not match the reformatted file name
            try:
                # Attempt to iterate through all the files until one is found
                # that matches the person in the attribute file. Keep track of
                # the indices of the files to be removed later.
                next_ = 0
                to_delete = []
                to_delete_names = []
                while attr['person'] != reformat(imgs[index + next_]):
                    to_delete.append(index + next_)
                    to_delete_names.append(reformat(imgs[index + next_]))
                    next_ += 1
            except:
                # If we reach the end of the files list without finding a
                # match, then we need to delete persons from the attributes
                # list until the indexed person is in the list of files that
                # would otherwise be deleted from the list constructed in the
                # `try` block above.
                next_ = 0
                while attrs[index + next_][1]['person'] not in to_delete_names:
                    del attrs[index + next_]
                    next_ += 1
            else:
                # If there's no exception (i.e. we find a match in the files
                # list), then go ahead and remove those files names from the
                # copied list.
                for index_to_delete in to_delete:
                    del imgs[index_to_delete]

    # Delete the files that are in the image directory but not in the filtered
    # list of images.
    for file_ in images:
        if file_ not in imgs:
            os.remove(data_dir + '/faces/' + file_)

    # Convert the filtered attributes list back into a dataframe and write to a
    # CSV file.
    df = pd.DataFrame(np.array([a[1] for a in attrs]), columns=lfw_attrs.columns)
    df = df.drop(columns=['person'])
    df.to_csv('new_data/edited_lfw_attributes.csv')

def create_dictionaries(data_dir):
    """
        Read in edited attributes file and create new dictionary numpy arrays
        and save them in the `dictionary` directory. These are to be used
        directly by the TF model.
    """

    attrs = pd.read_csv(data_dir + '/edited_lfw_attributes.csv', index_col=0)

    vocab = np.array(list(attrs.columns) + ['<PAD>', '<RARE>'])
    np.save('dictionary/vocab.npy', vocab)

    id2Word = np.array([[i, v] for i, v in enumerate(vocab)])
    np.save('dictionary/id2Word.npy', id2Word)

    word2Id = np.array([[v, i] for i, v in enumerate(vocab)])
    np.save('dictionary/word2Id.npy', word2Id)

def create_captions(data_dir, num_features):
    """
        Iterate over the edited attributes file and, for each row, create 10
        random variations of the num_features strongest attributes for the
        person. Save the result as `train_captions.npy` to be used directly by
        the TF model.
    """

    attrs = pd.read_csv(data_dir + '/edited_lfw_attributes.csv', index_col=0)
    word2Id = np.load('dictionary/word2Id.npy')

    captions = []
    for row in attrs.iterrows():

        # Sort to get the num_features strongest attributes
        sorted_row = row[1].sort_values(ascending=False)

        attributes, variations = [], []
        for col, val in sorted_row[:num_features].iteritems():
            attributes.append([x[1] for x in word2Id if x[0] == col][0])
        variations.append(attributes)

        # Randomly sample the attributes to create 9 variations
        for i in range(9):
            variations.append(sample(attributes, len(attributes)))

        # Add all 10 attribute variations to the list of "captions"
        captions.append(variations)

    np.save('train_captions.npy', captions)


def create_images(data_dir):
    """
        Iterate over all the trimmed image files and convert them into numpy
        arrays and append these to a list which is itself converted to a numpy
        array. The result of this is `train_images.npy` to be used directly by
        the TF model.
    """

    # Read in trimmed images and sort by file (person) names
    images = sorted(os.listdir(data_dir + '/faces'))

    # Convert PPM files to PIL Images, then from PIL Images to NumPy arrays
    np_images = []
    for img in images:
        pil_img = Image.open(data_dir + '/faces/' + img)
        np_images.append(np.array(pil_img))

    np.save('train_images.npy', np.array(np_images))


if __name__ == '__main__':
    DATA_DIR = '/home/aweeeezy/bin/school/deep-learning/text-to-image_synthesis/StackGAN-TensorFlow/new_data'

    #ensure_matching(DATA_DIR)
    #create_dictionaries(DATA_DIR)
    create_captions(DATA_DIR, 10)
    #create_images(DATA_DIR)
