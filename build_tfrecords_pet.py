"""minimal example showing how to generate TensorFlow TFRecord file."""

import tensorflow as tf
import os
#tf.enable_eager_execution()


# All raw values should be converted to a type compatible with tf.Example. Use
# the following functions to do these convertions.
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def write_record(IMAGE_PATH,OUTPUT_PATH,IMAGE, record_name, lab):
    os.chdir(IMAGE_PATH)
    # Read image raw data, which will be embedded in the record file later.
    image_string = open(IMAGE, 'rb').read()
    
    # Manually set the label to 0. This should be set according to your situation.
    label = lab 
    
    # For each sample there are two features: image raw data, and label. Wrap them in a single dict.
    feature = {
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_string),
    }
    
    # Create a `example` from the feature dict.
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    os.chdir(OUTPUT_PATH)
  
    # Write the serialized example to a record file.
    with tf.io.TFRecordWriter(record_name+'.tfrecords') as writer:
        writer.write(tf_example.SerializeToString())
    os.chdir('../')#get back into main directory


"""
def read_record(IMAGE):
    os.chdir(OUTPUT_PATH)
    # Use dataset API to import date directly from TFRecord file.
    raw_image_dataset = tf.data.TFRecordDataset(IMAGE[-8:-4]+'.tfrecords')

    # Create a dictionary describing the features. The key of the dict should be the same with the key in writing function.
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    
    # Define the parse function to extract a single example as a dict.
    def _parse_image_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    
    # If there are more than one example, use a for loop to read them out.
    for image_features in parsed_image_dataset:
        image_raw = image_features['image_raw'].numpy()
        label = image_features['label'].numpy()
"""
        
if __name__ == "__main__":
    OUTPUT_PATH = "../tfrecords" #relative path from where script is located to where output folder is
    lab = 0
    rootdir = "." #where all of your image data is located in numerically labeled folders
    for subdir, dirs, files in os.walk(rootdir):
        for file in files:
            if(len(subdir)>1 and (subdir[2] =='0' or subdir[2]=='1')):
#               print("image path ",IMAGE_PATH)
#               print("filename ",file)
#               print("label ",subdir[2])
                record_name = file[-8:-4]+'_'+subdir[2]
                write_record(subdir,OUTPUT_PATH, file, record_name,int(subdir[2]))
#               read_record(IMAGE_PATH)
