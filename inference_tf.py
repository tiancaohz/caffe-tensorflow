import tensorflow as tf
import cv2
import numpy as np

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


if __name__ == '__main__':
    graph = load_graph('./standalonehybrid.pb')

    x = graph.get_tensor_by_name('prefix/data:0')
    y = graph.get_tensor_by_name('prefix/prob:0')


    im = cv2.imread('/Users/tiancao/data/caffe_model/places365_vgg/12.jpg')

    WIDTH, HEIGHT = 224, 224
    im = cv2.resize(im, (WIDTH, HEIGHT))

    # Places was using batches of 10 images
    batch = np.array([im for i in range(10)])

    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={x: batch})

    print(y_out.argmax(axis=1))
